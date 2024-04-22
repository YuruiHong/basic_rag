# description: 测试用的客户端

import json
import random
import time
from multiprocessing import Pool

import numpy as np
import requests


def run_client(url, query):
    response = requests.post(url, json.dumps({"query": query}))
    return json.loads(response.text)


def cal_time_result(time_list):
    tp = np.percentile(time_list, [50, 90, 95, 99])
    print(
        f"tp50:{tp[0] * 1000:.4f}ms, tp90:{tp[1]* 1000:.4f}ms,tp95:{tp[2]* 1000:.4f}ms,tp99:{tp[3]* 1000:.4f}ms"
    )
    print(f"average: {sum(time_list) / len(time_list)}")
    print(f"qps:{len(time_list) / sum(time_list):.4f}")


def single_test(url, query_list, num, process_id=0):
    # query_list: 待请求query列表，num请求个数
    print(f"running process: process-{process_id}")
    time_list = []
    for i in range(num):
        start_time = time.time()
        query = random.choice(query_list)
        requests.post(url, json.dumps({"query": query}))
        end_time = time.time()
        time_list.append(end_time - start_time)
    return time_list


def batch_test(query_list, process_num, request_num):
    # query_list:待请求query列表，process_num进程个数，request_num请求个数(每个进程)
    pool = Pool(processes=process_num)
    process_result = []
    for i in range(process_num):
        process_result.append(
            pool.apply_async(
                single_test,
                args=(query_list, request_num, str(i)),
            )
        )
        # processes.append(Process(target=single_test, args=(query_list, request_num, str(i), )))

    pool.close()
    pool.join()

    time_list = []
    for result in process_result:
        time_list.extend(result.get())
    return time_list


# response = requests.post("http://127.0.0.1:9090/a", json.dumps({"query": "你好啊1"}))
# print(json.loads(response.text))

# response = requests.post("http://127.0.0.1:9091/b", json.dumps({"query": "你好啊2"}))
# print(json.loads(response.text))

if __name__ == "__main__":
    from loguru import logger

    # url = "http://127.0.0.1:9090/searcher"
    # logger.info(run_client(url, "什么人不能吃花生")) # 单元测试
    # url = "http://127.0.0.1:9092/llm_model"
    # logger.info(run_client(url, "什么人不能吃花生")) # 单元测试
    url = "http://127.0.0.1:9093/dialogue_manager"
    logger.info(run_client(url, "什么人不能吃花生"))  # 单元测试

    # time_list = [0]
    # time_list = single_test(url, ["你好啊","今天天气怎么样"], 100) # 批量单进程测试
    # cal_time_result(time_list=time_list)
    # time_list = batch_test(url, ["你好啊","今天天气怎么样"], 4, 100) # 多进程压测
    # cal_time_result(time_list=time_list)
