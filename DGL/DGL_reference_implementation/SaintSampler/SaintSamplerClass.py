from dgl.dataloading import SAINTSampler

class SAINTSampler_custom(SAINTSampler):
    def original_method(self, g, indices):
        """
        Keeping the indices of nodes intact
        """
        node_ids = self.sampler(g)
        # relabel_nodes=False
        sg = g.subgraph(node_ids, relabel_nodes=False, output_device=self.output_device)
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg

