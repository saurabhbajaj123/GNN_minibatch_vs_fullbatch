import torch
import torch.nn.functional as F

def train(epoch, model, train_loader, train_idx, device, x, y):
    model.train()
    #pbar = tqdm(total=train_idx.size(0))
    #pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()    
        l1_emb, l2_emb, l3_emb = model(x[n_id], adjs)
        #print("Layer 1 embeddings", l1_emb.shape)
        #print("Layer 2 embeddings", l1_emb.shape)
        out = l3_emb.log_softmax(dim=-1)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        #pbar.update(batch_size)

    #pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc