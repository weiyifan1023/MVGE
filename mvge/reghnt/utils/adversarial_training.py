import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='robertaembeddings.word_embeddings_layer.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                print('fgm attack')
                #这里加入fgm attack来判断是否进行对抗训练了
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='robertaembeddings.word_embeddings_layer.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                print('fgm restore')
                #这里加入fgm restore判断是否恢复参数了
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
