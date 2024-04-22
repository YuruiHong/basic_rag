import json
import os

import faiss


class VecIndex:
    def __init__(self, index_dim) -> None:
        description = "HNSW64"
        measure = faiss.METRIC_L2
        self.index = faiss.index_factory(index_dim, description, measure)

    def insert(self, vec):
        self.index.add(vec)

    def batch_insert(self, vecs):
        self.index.add(vecs)

    def load(self, read_path):
        self.index = faiss.read_index(read_path)

    def save(self, save_path):
        faiss.write_index(self.index, save_path)

    def search(self, vec, num):
        # id, distance
        return self.index.search(vec, num)


class VecSearcher:
    def __init__(self):
        self.INDEX_FOLDER_PATH_TEMPLATE = "data/index/{}"

    def build(self, index_dim, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)
        if not os.path.exists(self.index_folder_path) or not os.path.isdir(
            self.index_folder_path
        ):
            os.mkdir(self.index_folder_path)

        self.invert_index = VecIndex(index_dim)
        self.forward_index = []  # 检索正排，实质上只是个list，通过ID获取对应的内容

    def insert(self, vec, doc):
        self.invert_index.insert(vec)
        # self.invert_index.batch_insert(vecs)

        self.forward_index.append(doc)

    def save(self):
        with open(
            self.index_folder_path + "/forward_index.txt", "w", encoding="utf8"
        ) as f:
            for data in self.forward_index:
                f.write(f"{json.dumps(data, ensure_ascii=False)}\n")

        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")

    def load(self, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)

        self.invert_index = VecIndex()
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")

        self.forward_index = []
        with open(self.index_folder_path + "/forward_index.txt", encoding="utf8") as f:
            for line in f:
                self.forward_index.append(json.loads(line.strip()))

    def search(self, vecs, nums=5):
        search_res = self.invert_index.search(vecs, nums)
        recall_list = []
        for idx in range(nums):
            # recall_list_idx, recall_list_detail, distance
            recall_list.append(
                [
                    search_res[1][0][idx],
                    self.forward_index[search_res[1][0][idx]],
                    search_res[0][0][idx],
                ]
            )
        # recall_list = list(filter(lambda x: x[2] < 100, result))

        return recall_list
