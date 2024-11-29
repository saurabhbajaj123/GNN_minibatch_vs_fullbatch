import torch.nn.functional as F
from module.model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score

import random
import wandb
import psutil
from datetime import timedelta

from pthflops import count_ops
from torch.profiler import profile, record_function, ProfilerActivity

def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


@torch.no_grad()
def evaluate_induc(name, model, g, mode, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, loss_fcn, result_file_name=None):
    model.eval()
    model.cpu()

    device = next(model.parameters()).device
    # print(device)
    # g = g.to(device)
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    train_mask = g.ndata['train_mask']
    logits = model(g, feat)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    train_logits, train_labels = logits[train_mask], labels[train_mask]
    val_acc = calc_acc(val_logits, val_labels)
    test_acc = calc_acc(test_logits, test_labels)
    train_acc = calc_acc(train_logits, train_labels)
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    val_loss = loss_fcn(val_logits, val_labels)
    val_loss = val_loss.item() / len(val_labels)
    test_loss = loss_fcn(test_logits, test_labels)
    test_loss = test_loss.item() / len(test_labels)
    train_loss = loss_fcn(train_logits, train_labels)
    train_loss = train_loss.item() / len(train_labels)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, val_acc, test_acc, train_acc, val_loss, test_loss, train_loss


def average_gradients(model, n_train):
    reduce_time = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        t0 = time.time()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= n_train
        reduce_time += time.time() - t0
    return reduce_time


def move_to_cuda(graph, part, node_dict):

    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'))
    part = part.int().to(torch.device('cuda'))

    return graph, part, node_dict


def get_pos(node_dict, gpb):
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            part_size = gpb.partid2nids(i).shape[0]
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, 'cuda')
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos


def get_recv_shape(node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape


def create_inner_graph(graph, node_dict):
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))


def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)
    return construct(part, graph, pos, one_hops)


def move_train_first(graph, node_dict, boundary):
    train_mask = node_dict['train_mask']
    num_train = torch.count_nonzero(train_mask).item()
    num_tot = graph.num_nodes('_V')

    new_id = torch.zeros(num_tot, dtype=torch.int, device='cuda')
    new_id[train_mask] = torch.arange(num_train, dtype=torch.int, device='cuda')
    new_id[torch.logical_not(train_mask)] = torch.arange(num_train, num_tot, dtype=torch.int, device='cuda')

    u, v = graph.edges()
    u[u < num_tot] = new_id[u[u < num_tot].long()]
    v = new_id[v.long()]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    for key in node_dict:
        node_dict[key][new_id.long()] = node_dict[key][0:num_tot].clone()

    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()

    return graph, node_dict, boundary


def create_graph_train(graph, node_dict):
    u, v = graph.edges()
    num_u = graph.num_nodes('_U')
    sel = nonzero_idx(node_dict['train_mask'][v.long()])
    u, v = u[sel], v[sel]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if graph.num_nodes('_U') < num_u:
        graph.add_nodes(num_u - graph.num_nodes('_U'), ntype='_U')
    return graph, node_dict['in_degree'][node_dict['train_mask']]


def precompute(graph, node_dict, boundary, recv_shape, args):
    rank, size = dist.get_rank(), dist.get_world_size()
    in_size = node_dict['inner_node'].bool().sum()
    feat = node_dict['feat']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
        else:
            send_info.append(feat[b])
    recv_feat = data_transfer(send_info, recv_shape, args.backend, dtype=torch.float)
    if args.model == 'graphsage':
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph['_E'].update_all(fn.copy_u('h', 'm'),
                                   fn.sum(msg='m', out='h'),
                                   etype='_E')
            mean_feat = graph.nodes['_V'].data['h'] / node_dict['in_degree'][0:in_size].unsqueeze(1)
        return torch.cat([feat, mean_feat[0:in_size]], dim=1)
    else:
        raise Exception


def create_model(layer_size, args):
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, args.use_pp, norm=args.norm, dropout=args.dropout,
                         n_linear=args.n_linear, train_size=args.n_train)
    else:
        raise NotImplementedError


def reduce_hook(param, name, n_train):
    def fn(grad):
        ctx.reducer.reduce(param, name, grad, n_train)
    return fn


def construct(part, graph, pos, one_hops):
    # print(f"part = {part}, graph = {graph}, pos = {pos}, one_hops = {one_hops}")
    rank, size = dist.get_rank(), dist.get_world_size()
    # print(f"part.num_nodes = {part.num_nodes()}, graph.num_nodes = {graph.num_nodes()}")
    tot = part.num_nodes()
    u, v = part.edges()
    # print(f"u = {u}, v = {v}")
    u_list, v_list = [u], [v]
    # print(f"len(u) = {len(u)}, len(v_list) = {len(v)}")
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    # print(f"u = {u}, v = {v}")
    # print(f"len(u) = {len(u)}, len(v) = {len(v)}")
    # print(f"len(torch.unique(u)) = {len(torch.unique(u))}, len(torch.unique(v)) = {len(torch.unique(v))}")
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    # print(f"g in construct = {g}")
    # print(f"g.num_nodes() bafore adding new nodes = {g.num_nodes()}")
    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    # if g.num_nodes('_V') < part.num_nodes():
    #     g.add_nodes(tot - g.num_nodes('_V'), ntype='_V')
    # print(f"g.num_nodes() after construct = {g.num_nodes()}")
    return g


def extract(graph, node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    sel = (node_dict['part_id'] < size)
    for key in node_dict.keys():
        if node_dict[key].shape[0] == sel.shape[0]:
            node_dict[key] = node_dict[key][sel]
    graph = dgl.node_subgraph(graph, sel, store_ids=False)
    return graph, node_dict


def run(graph, node_dict, gpb, args):
    
    rank, size = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        wandb.init(
            project="PipeGCN-{}-{}".format(args.dataset, args.model),
            name=f"enable_pipeline-{args.enable_pipeline}, n_hidden-{args.n_hidden}, n_layers-{args.n_layers}",
            # notes="HPO by varying only the n_hidden and n_layers"
        # project="PipeGCN-{}-{}".format(args.dataset, args.model),
        )

        wandb.log({
            'torch_seed': torch.initial_seed() & ((1<<63)-1) ,
        })
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if rank == 0 and args.eval:
        if args.dataset_subgraph_path == '':
            full_g, n_feat, n_class = load_data(args.dataset)
        else:
            full_g, n_feat, n_class = load_subgraph(args.dataset_subgraph_path)
        # full_g, n_feat, n_class = load_data(args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            # val_g, test_g = full_g.clone(), full_g.clone()
            val_g, test_g = full_g, full_g
        # del full_g

    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/', exist_ok=True)

    part = create_inner_graph(graph.clone(), node_dict)
    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()

    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges '
          f'{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')

    graph, part, node_dict = move_to_cuda(graph, part, node_dict)
    boundary = get_boundary(node_dict, gpb)
    # print(f"boundary = {boundary}")
    layer_size = get_layer_size(args.n_feat, args.n_hidden, args.n_class, args.n_layers)
    # print(f"layer_size = {layer_size}")
    pos = get_pos(node_dict, gpb)
    # print(f"pos = {pos}")
    graph = order_graph(part, graph, gpb, node_dict, pos)
    # print(f"graph = {graph}")
    in_deg = node_dict['in_degree']
    # print(f"in_deg = {in_deg}, rank = {rank}")
    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary)

    recv_shape = get_recv_shape(node_dict)

    ctx.buffer.init_buffer(num_in, graph.num_nodes('_U'), boundary, recv_shape, layer_size[:args.n_layers-args.n_linear],
                           use_pp=args.use_pp, backend=args.backend, pipeline=args.enable_pipeline,
                           corr_feat=args.feat_corr, corr_grad=args.grad_corr, corr_momentum=args.corr_momentum)

    if args.use_pp:
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)

    labels = node_dict['label'][node_dict['train_mask']]
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()

    del boundary
    del part
    del pos

    torch.manual_seed(args.seed)
    model = create_model(layer_size, args)
    model.cuda()

    ctx.reducer.init(model)

    for i, (name, param) in enumerate(model.named_parameters()):
        param.register_hook(reduce_hook(param, name, args.n_train))

    best_model, best_val_acc, best_test_acc, best_train_acc = None, 0, 0, 0

    if args.grad_corr and args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_grad_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.grad_corr:
        result_file_name = 'results/%s_n%d_p%d_grad.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    else:
        result_file_name = 'results/%s_n%d_p%d.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_dur, comm_dur, reduce_dur = [], [], []
    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    feat = node_dict['feat']

    node_dict.pop('train_mask')
    node_dict.pop('inner_node')
    # node_dict.pop('part_id')
    node_dict.pop(dgl.NID)

    if not args.eval:
        node_dict.pop('val_mask')
        node_dict.pop('test_mask')
    
    # prev_loss = float('inf')
    train_time = 0
    # loss_list = []
    val_acc = 0
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_train_loss = float('inf')
    no_improvement_count = 0

    # print_memory('before epoch start')
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        # print_memory('before model call')

        if args.model == 'graphsage':

            # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
            #     with record_function("model_inference"):
            # print(graph.shape, feat.shape, in_deg.shape)
            # inp = (graph, feat, in_deg)
            # count_ops(model, inp)
            # print(feat.shape)
            # dist.barrier()
            logits, flops = model(graph, feat, in_deg)
            dist.barrier()
            print(f"rank = {rank}")
            print(f"final flops = {flops}")
            dist.all_reduce(flops,  op=dist.ReduceOp.SUM)
            if rank == 0: 
                print(f"final flops = {flops}")

                data = {
                    'n_layers': [args.n_layers],
                    'n_hidden': [args.n_hidden],
                    'n_gpus': [args.n_partitions],
                    'total_flops': [flops.item()]
                    # 'total_dst_nodes': [total_dst_nodes.item()], 
                    # 'total_edges': [total_edges.item()]

                }
                df = pd.DataFrame(data)
                print(df)
                file_path = f'/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/{args.dataset}_gcn_flops.csv'
                try:
                    df.to_csv(file_path, mode='a', index=False, header=False)
                except Exception as e:
                    print(e)


        else:
            raise Exception
        if args.inductive:
            loss = loss_fcn(logits, labels)
        else:
            loss = loss_fcn(logits[train_mask], labels)
        del logits
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        ctx.buffer.next_epoch()

        pre_reduce = time.time()
        ctx.reducer.synchronize()
        reduce_time = time.time() - pre_reduce
        optimizer.step()

        peak_mem = print_memory('after optimizer step')

        # peak_mem = torch.tensor(peak_mem)
        # dist.barrier()
        # dist.all_reduce(peak_mem, op=dist.ReduceOp.SUM)
        # dist.barrier()
        
        # if rank == 0: 
        #     print(f"final flops = {flops}")

        #     data = {
        #         'n_layers': [args.n_layers],
        #         'n_hidden': [args.n_hidden],
        #         'n_gpus': [args.n_partitions],
        #         'peak_mem': [peak_mem]
        #         # 'total_dst_nodes': [total_dst_nodes.item()], 
        #         # 'total_edges': [total_edges.item()]

        #     }
        #     df = pd.DataFrame(data)
        #     print(df)
        #     file_path = f'/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/{args.dataset}_mem.csv'
        #     try:
        #         df.to_csv(file_path, mode='a', index=False, header=False)
        #     except Exception as e:
        #         print(e)
        
        t1 = time.time()
        train_time += t1 - t0
        # print(f"Epoch time  = {t1-t0}")
        # avg_loss = loss.item() / len(labels)
        # loss_list.append(avg_loss)
        if epoch % args.log_every != 0:
            train_dur.append(time.time() - t0)
            comm_dur.append(ctx.comm_timer.tot_time())
            reduce_dur.append(reduce_time)

        if (epoch + 1) % args.log_every == 0:
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                  rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), loss.item() / part_train))

        ctx.comm_timer.clear()

        del loss
        
        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:

            # print(prof.key_averages().table(row_limit=20))
            # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

            # print_memory(f'Epoch = {epoch}')
            if thread is not None:
                if args.inductive:
                    model_copy, val_acc = thread.get()
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = model_copy
                    wandb.log({'val_acc': val_acc,
                        # 'test_acc': test_acc,
                        # 'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                    })
                else:
                    model_copy, val_acc, test_acc, train_acc, val_loss, test_loss, train_loss = thread.get()
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        best_train_acc = train_acc
                        best_model = model_copy
                        no_improvement_count = 0
                    else:
                        no_improvement_count += args.log_every
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_test_loss = test_loss
                        best_train_loss = train_loss


                    wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        'train_time': train_time,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'best_val_loss': best_val_loss,
                        'best_test_loss': best_test_loss,
                        'best_train_loss': best_train_loss,

                    })

                    
    
                    
            model_copy = copy.deepcopy(model)
            if not args.inductive:
                thread = pool.apply_async(evaluate_trans, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, loss_fcn, result_file_name))
            else:
                thread = pool.apply_async(evaluate_induc, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, 'val', result_file_name))
        
        dist.barrier()
        break_condition = False
        if epoch > 50 and no_improvement_count >= args.patience:
            break_condition = True
        dist.barrier()
        # print(np.mean(loss_list[-11:-1]), avg_loss)
        break_condition_tensor = torch.tensor(int(break_condition)).cuda()
        dist.all_reduce(break_condition_tensor, op=dist.ReduceOp.BOR)
        break_condition = bool(break_condition_tensor.item())
        # print(break_condition)
        if break_condition:
            print(f'Early stopping after {epoch + 1} epochs.')
            # break

        # prev_loss = avg_loss

    print_memory("after all epochs")
    if args.eval and rank == 0:
        if thread is not None:
            if args.inductive:
                model_copy, val_acc = thread.get()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model_copy
            else:
                model_copy, val_acc, test_acc, train_acc, val_loss, test_loss, train_loss = thread.get()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                    best_model = model_copy
                

        # torch.save(best_model.state_dict(), 'model/' + args.graph_name + '_final.pth.tar')
        # print('model saved')
        print("Validation accuracy {:.2%}".format(best_val_acc))
        # _, final_test_acc = evaluate_induc('Test Result', best_model, test_g, 'test')
        wandb.log({'val_acc': val_acc,
                'test_acc': test_acc,
                'train_acc': train_acc,
                'best_test_acc': best_test_acc,
                'best_val_acc': best_val_acc,
                'best_train_acc': best_train_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                # 'final_test_acc': final_test_acc,
        })
    
    # total_training_time_tensor = torch.tensor(np.sum(train_dur)).to(rank)
    # dist.all_reduce(total_training_time_tensor, op=dist.ReduceOp.SUM)
    # total_training_time = total_training_time_tensor.item()
    # print(f"total_training_time = {total_training_time}")
    if rank == 0:
        wandb.log({
            'torch_seed': torch.initial_seed(),
            'total train time per GPU': np.sum(train_dur),
            'total train time': np.sum(train_dur) * args.n_partitions,
            'train time per epoch': (np.sum(train_dur) * args.n_partitions) / (epoch+1),
            'num epochs': (epoch+1),
            'average train time per epoch': np.mean(train_dur),
        })

def check_parser(args):
    if args.norm == 'none':
        args.norm = None

def print_memory(s):
    torch.cuda.synchronize()
    print(f"cpu memory  = {psutil.virtual_memory()}")
    print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024
    ))
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def init_processes(rank, size, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    # dist.init_process_group(args.backend, rank=rank, world_size=size)
    dist.init_process_group(args.backend, rank=rank, world_size=size, timeout=timedelta(days=1))
    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    g, node_dict, gpb = load_partition(args, rank)
    # print(g, node_dict, gpb)
    # print_memory('before run')
    run(g, node_dict, gpb, args)
