import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns

class M_PositionalEncoder(nn.Module):
    """
    位置编码
    """
    def __init__(self,d_model,max_seq_len=512) -> None:
        super(M_PositionalEncoder,self).__init__()
        self.d_model = d_model
        pe = torch.zeros(size=(max_seq_len,self.d_model))
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos,i] = math.sin(pos / math.pow(10000,2*i/self.d_model))
                pe[pos,i+1] = math.cos(pos / math.pow(10000,2*i/self.d_model))
        # print('pe.shape:', pe.shape)
        self.register_buffer('pe', pe)

    def show_pe_matrix(self):
        # sns.heatmap(self.pe, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Color scale'})
        plt.imshow(self.pe)
        plt.title('cos&sin postional embedding')
        plt.savefig('Positional Embedding')
        plt.show()

    def forward(self, x):
        seq_len = x.size(0)
        # print('seq_len:', seq_len)
        x = x + self.pe[:seq_len,:]
        return x


if __name__ == "__main__":
    x = torch.zeros(120,2048)
    m_PositionalEncoder = M_PositionalEncoder(2048,512)
    vector_with_pos = m_PositionalEncoder(x)
    print('vector_with_pos.shape:', vector_with_pos.shape)
    m_PositionalEncoder.show_pe_matrix()
