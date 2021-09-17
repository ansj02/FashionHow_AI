from data_io import *
from model import *
from ast import literal_eval

class mevac():
    def __init__(self, args):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.emb_size = args.emb_size
        self.img_size = args.img_size
        self.img_emb_size = args.img_emb_size
        self.key_size = args.key_size
        self.res_size = args.res_size
        self.hops = args.hops
        self.embedding_model = load_embedding_model()
        self.story_size = args.max_story_size
        self.sentence_len = args.max_sentence_len
        self.learning_rate = args.learning_rate
        self.anti_zero = args.anti_zero
        self.random_seed = args.random_seed
        self.path = '.'
        torch.manual_seed(self.random_seed)

    def train_set(self):
        dia_index_list, dia_code_list, dia_data_list, image_list, dia_tag_list = get_data_list('train')
        dia_emb_list = get_embedding_data(dia_data_list, self.sentence_len, self.emb_size, self.embedding_model)

        train_querySet, train_storySet, train_imageSet = new_get_train_dataSet(dia_index_list, dia_code_list,
                                                                               dia_emb_list,
                                                                               image_list, dia_tag_list,
                                                                               self.story_size, self.sentence_len,
                                                                               self.emb_size)
        num_data = len(train_querySet)

        train_querySet = np.array(train_querySet, dtype=np.float32)
        train_storySet = np.array(train_storySet, dtype=np.float32)
        train_querySet = torch.tensor(train_querySet)
        train_storySet = torch.tensor(train_storySet)
        return train_querySet, train_storySet, train_querySet, train_storySet, train_imageSet, num_data

    def img_emb_train(self):
        train_querySet, train_storySet, train_querySet, train_storySet, train_imageSet, num_data = self.train_set()

        img_emb_model = ImageEmbeddingModel(self.emb_size, self.img_size)
        optimizer = torch.optim.SGD(img_emb_model.parameters(), lr=self.learning_rate/1e+4, momentum=0.9)

        train_querySet = np.array(train_querySet, dtype=np.float32)
        train_querySet = torch.tensor(train_querySet)

        imageSet = np.zeros((num_data, 4, 3, self.img_size[0], self.img_size[1]), dtype=np.float32)

        for img_id, img in enumerate(train_imageSet):
            for idx in range(4):
                imageSet[img_id, idx] = np.array(read_padded_image(img[idx]), dtype=np.float32)

        for i in range(int(num_data / self.batch_size)):
            batch_querySet = train_querySet[i * self.batch_size:(i + 1) * self.batch_size]
            batch_imageSet = imageSet[i * self.batch_size:(i + 1) * self.batch_size]
            batch_imageSet = torch.tensor(batch_imageSet).reshape(-1, 3, self.img_size[0], self.img_size[1])

            batch_querySet = batch_querySet.reshape(self.batch_size, self.sentence_len, self.emb_size)
            batch_querySet = torch.sum(batch_querySet, 1)

            for _ in range(int(self.epoch/2)):
                emb_img = img_emb_model(batch_imageSet)  # 80,300
                emb_img = emb_img.reshape(self.batch_size, 4, self.emb_size)
                emb_img = torch.sum(emb_img, 1)
                loss = torch.sum((emb_img - batch_querySet) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss)
        torch.save(img_emb_model.state_dict(), self.path+'/model_file/img_emb_model_state_dict')

        meta_data_set = pd.read_csv(self.path+'/model_file/metaDataSet.csv', converters={'character_emb': literal_eval})
        make_img_emb_file(self.emb_size, self.img_size, meta_data_set)

    def train(self):
        train_querySet, train_storySet, train_querySet, train_storySet, train_imageSet, num_data = self.train_set()

        dec_model = DecisionModel(self.res_size, self.emb_size, self.img_emb_size, self.key_size,
                                   self.story_size, self.sentence_len, self.hops, self.batch_size)
        optimizer = torch.optim.SGD(dec_model.parameters(), lr=self.learning_rate, momentum=0.9)

        with open(self.path+'/model_file/imgEmbDataSet.json') as json_file:
            img_emb_data = json.load(json_file)
        embImgSet = np.zeros((num_data, 4, int(self.img_emb_size/self.emb_size), self.emb_size), dtype=np.float32)
        for set_id, imgSet in enumerate(train_imageSet):
            for img_id, img in enumerate(imgSet):
                if img:
                    embImgSet[set_id, img_id] = np.array(img_emb_data[img + '.jpg'], dtype=np.float32)
        embImgSet = np.sum(embImgSet, 1)
        embImgSet = embImgSet.reshape((num_data, self.img_emb_size))

        for i in range(int(num_data / self.batch_size)):
            batch_querySet = train_querySet[i * self.batch_size:(i + 1) * self.batch_size]
            batch_storySet = train_storySet[i * self.batch_size:(i + 1) * self.batch_size]
            batch_imageSet = embImgSet[i * self.batch_size:(i + 1) * self.batch_size]
            batch_imageSet = torch.tensor(batch_imageSet)

            for _ in range(self.epoch):
                dia_res, img_res = dec_model(batch_storySet, batch_querySet, batch_imageSet)
                dia_res = dia_res.reshape((self.batch_size, self.res_size))
                loss = (torch.sum((img_res - dia_res) ** 2) + (
                            torch.exp(-1 * torch.sum((torch.zeros(self.batch_size, self.res_size) - img_res) ** 2)) +
                            torch.exp(-1 * torch.sum((torch.zeros(self.batch_size,
                                                                  self.res_size) - dia_res) ** 2))) * self.anti_zero) / self.batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss)
        torch.save(dec_model.state_dict(), self.path+'/model_file/dec_model_state_dict')

    def test_loss(self):
        dia_index_list, dia_code_list, dia_data_list, image_list, dia_tag_list = get_data_list('test')
        dia_emb_list = get_embedding_data(dia_data_list, self.sentence_len, self.emb_size, self.embedding_model)

        test_querySet, test_storySet, test_imageSet = new_get_test_dataSet(dia_index_list, dia_code_list, dia_emb_list,
                                                                           image_list, self.story_size,
                                                                           self.sentence_len, self.emb_size)
        num_data = len(test_querySet)

        test_querySet = np.array(test_querySet, dtype=np.float32)
        test_storySet = np.array(test_storySet, dtype=np.float32)
        test_querySet = torch.tensor(test_querySet)
        test_storySet = torch.tensor(test_storySet)

        dec_model = DecisionModel(self.res_size, self.emb_size, self.img_emb_size, self.key_size,
                                  self.story_size, self.sentence_len, self.hops, num_data)
        dec_model.load_state_dict(torch.load(self.path + '/model_file/dec_model_state_dict'))

        with open(self.path + '/model_file/imgEmbDataSet.json') as json_file:
            img_emb_data = json.load(json_file)
        embImgSet = np.zeros((num_data, 3, 4, int(self.img_emb_size / self.emb_size), self.emb_size), dtype=np.float32)
        for set_id, imgListSet in enumerate(test_imageSet):
            for list_id, imgList in enumerate(imgListSet):
                for img_id, img in enumerate(imgList):
                    if img:
                        embImgSet[set_id, list_id, img_id] = np.array(img_emb_data[img + '.jpg'], dtype=np.float32)
        embImgSet = np.sum(embImgSet, 2)
        embImgSet = embImgSet.reshape((num_data, 3, self.img_emb_size))
        embImgSet1 = torch.tensor(embImgSet[:, 0, :])
        embImgSet2 = torch.tensor(embImgSet[:, 1, :])
        embImgSet3 = torch.tensor(embImgSet[:, 2, :])

        dia_res, img_res = dec_model(test_storySet, test_querySet, embImgSet1)
        dia_res = dia_res.reshape((num_data, self.res_size))
        loss1 = ((img_res - dia_res) ** 2).tolist()
        dia_res, img_res = dec_model(test_storySet, test_querySet, embImgSet2)
        dia_res = dia_res.reshape((num_data, self.res_size))
        loss2 = ((img_res - dia_res) ** 2).tolist()
        dia_res, img_res = dec_model(test_storySet, test_querySet, embImgSet3)
        dia_res = dia_res.reshape((num_data, self.res_size))
        loss3 = ((img_res - dia_res) ** 2).tolist()
        return loss1, loss2, loss3, num_data

    def test(self):
        loss1, loss2, loss3, num_data = self.test_loss()

        score = 0

        for idx in range(num_data):
            if min(loss1[idx], loss2[idx], loss3[idx]) == loss1[idx]:
                if loss2[idx] < loss3[idx]:
                    score = score + 3
                else:
                    score = score + 2
            elif min(loss1[idx], loss2[idx], loss3[idx]) == loss2[idx]:
                if loss1[idx] < loss3[idx]:
                    score = score + 1
                else:
                    score = score - 1
            elif min(loss1[idx], loss2[idx], loss3[idx]) == loss3[idx]:
                if loss1[idx] < loss2[idx]:
                    score = score - 2
                else:
                    score = score - 3
        print(score)

    def predict(self):
        loss1, loss2, loss3, num_data = self.test_loss()

        pred_list = np.zeros(num_data)

        for idx in range(num_data):
            if min(loss1[idx], loss2[idx], loss3[idx]) == loss1[idx]:
                if loss2[idx] < loss3[idx]:
                    pred_list[idx] = 0
                else:
                    pred_list[idx] = 1
            elif min(loss1[idx], loss2[idx], loss3[idx]) == loss2[idx]:
                if loss1[idx] < loss3[idx]:
                    pred_list[idx] = 2
                else:
                    pred_list[idx] = 3
            elif min(loss1[idx], loss2[idx], loss3[idx]) == loss3[idx]:
                if loss1[idx] < loss2[idx]:
                    pred_list[idx] = 4
                else:
                    pred_list[idx] = 5
        np.savetxt(self.path+"/prediction.csv", pred_list.astype(int), encoding='utf8', fmt='%d')
