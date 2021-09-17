import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEmbeddingModel(nn.Module):
    def __init__(self, img_emb_size, img_size):
        super().__init__()
        self.img_emb_size = img_emb_size
        self.mid_emb = img_emb_size*2
        self.kernel_size = 11
        self.drop_prob = 0.5
        self.emb_filter1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                                   out_channels=16,
                                                   kernel_size=self.kernel_size,
                                                   stride=1, padding=int(self.kernel_size/2)),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=int(self.kernel_size/2),
                                                      stride=int(self.kernel_size/2), padding=0))
        self.emb_filter2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                                   out_channels=32,
                                                   kernel_size=self.kernel_size,
                                                   stride=1, padding=int(self.kernel_size/2)),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=int(self.kernel_size / 2),
                                                      stride=int(self.kernel_size / 2), padding=0))

        tot_num = 16*16*32
        self.emb_linear1 = nn.Linear(tot_num, self.mid_emb, bias=True)
        nn.init.xavier_uniform_(self.emb_linear1.weight)
        self.emb_linear2 = nn.Linear(self.mid_emb, self.img_emb_size, bias=True)
        nn.init.xavier_uniform_(self.emb_linear2.weight)
        self.emb_linear = nn.Sequential(self.emb_linear1, nn.ReLU(),
                                        nn.Dropout(p=self.drop_prob),
                                        self.emb_linear2)

    def forward(self, img):
        emb_img = self.emb_filter1(img)
        emb_img = self.emb_filter2(emb_img)
        emb_img = emb_img.view(emb_img.size(0), -1)
        emb_img = self.emb_linear(emb_img)
        return emb_img


class MemN2N(nn.Module):
    def __init__(self, emb_size, key_size, story_size, sentence_len, hops, batch_size):
        super().__init__()
        self.emb_size = emb_size
        self.sentence_len = sentence_len
        self.story_size = story_size
        self.d = emb_size*2
        self.key_size = key_size
        self.hops = hops
        self.batch_size = batch_size

        self.A = nn.Parameter(torch.normal(mean=0.0, std=0.01,
                              size=(self.hops, self.emb_size, self.d)),
                              requires_grad=True)
        self.B = nn.Parameter(torch.normal(mean=0.0, std=0.01,
                              size=(emb_size, self.d)),
                              requires_grad=True)
        self.C = nn.Parameter(torch.normal(mean=0.0, std=0.01,
                              size=(self.hops, self.emb_size, self.d)),
                              requires_grad=True)
        self.H = nn.Parameter(torch.normal(mean=0.0, std=0.01,
                              size=(self.d, self.d)),
                              requires_grad=True)
        self.W = nn.Parameter(torch.normal(mean=0.0, std=0.01,
                              size=(self.d, self.key_size)),
                              requires_grad=True)

    def forward(self, stories, queries):
        u = torch.matmul(queries, self.B)
        u = torch.sum(u, 2)

        for i in range(self.hops):
            m_in = torch.matmul(stories, self.A[i])
            m_in = torch.reshape(m_in, (self.batch_size, self.story_size, self.sentence_len, self.d))
            m_in = torch.sum(m_in, 1)
            c = torch.matmul(stories, self.C[i])
            c = torch.reshape(c, (self.batch_size, self.story_size, self.sentence_len, self.d))
            c = torch.sum(c, 1)

            p = F.softmax(torch.matmul(u, torch.transpose(m_in, 1, 2)))
            o = torch.matmul(p, c)

            u = torch.matmul(u, self.H)+o
        a_h = F.softmax(torch.matmul(u, self.W))
        return a_h


class DecisionModel(nn.Module):
    def __init__(self, res_size, emb_size, img_emb_size, key_size, story_size, sentence_len, hops, batch_size):
        super().__init__()
        self.memory = MemN2N(emb_size, key_size, story_size, sentence_len, hops, batch_size)
        self.res_size = res_size
        self.dia_in = key_size
        self.dia_h1 = key_size * 2
        self.dia_h2 = key_size * 4
        self.dia_h3 = res_size * 4
        self.dia_h4 = res_size * 2
        self.img_in = img_emb_size
        self.img_h1 = img_emb_size * 2
        self.img_h2 = img_emb_size * 4
        self.img_h3 = res_size * 4
        self.img_h4 = res_size * 2

        self.dia_proc_layer = nn.Sequential(nn.Linear(self.dia_in, self.dia_h1), nn.ReLU(),
                                            nn.Linear(self.dia_h1, self.dia_h2), nn.ReLU(),
                                            nn.Linear(self.dia_h2, self.dia_h3), nn.ReLU(),
                                            nn.Linear(self.dia_h3, self.dia_h4), nn.ReLU(),
                                            nn.Linear(self.dia_h4, self.res_size))
        self.img_proc_layer = nn.Sequential(nn.Linear(self.img_in, self.img_h1), nn.ReLU(),
                                            nn.Linear(self.img_h1, self.img_h2), nn.ReLU(),
                                            nn.Linear(self.img_h2, self.img_h3), nn.ReLU(),
                                            nn.Linear(self.img_h3, self.img_h4), nn.ReLU(),
                                            nn.Linear(self.img_h4, self.res_size))

    def forward(self, stories, queries, img):
        dia_src = self.memory(stories, queries)
        dia_res = self.dia_proc_layer(dia_src)
        img_res = self.img_proc_layer(img)
        return dia_res, img_res

    def loss(self, dia_res, img_res):
        return torch.sum((dia_res-img_res)**2)/len(dia_res)