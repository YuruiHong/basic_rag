import copy

from dialogue_manager import DialogueManager
from llm_model import LlmModel
from loguru import logger
from searcher import Searcher
from tornado.escape import json_decode
from tornado.web import RequestHandler
from vec_model import VectorizeModel


class DialogueManagerHandler(RequestHandler):
    # description: 对话管理handler
    def initialize(self, dialogue_manager: DialogueManager):
        self.dialogue_manager = dialogue_manager
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":"XXX"
        # }

    async def post(self):
        answer = self.dialogue_manager.predict(
            json_decode(self.request.body).get("query", "")
        )
        logger.info(answer)
        response_body = {"answer": answer}
        self.write(response_body)


class LlmHandler(RequestHandler):
    # description: 大模型服务handler

    def initialize(self, llm_model: LlmModel):
        self.llm_model = llm_model
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":"XXX"
        # }

    async def post(self):
        logger.info(
            "request: {}".format(json_decode(self.request.body).get("query", ""))
        )
        answer = self.llm_model.predict(json_decode(self.request.body).get("query", ""))
        response_body = {"answer": answer[0]}
        logger.info(f"response: {response_body}")

        self.write(response_body)


class SearcherHandler(RequestHandler):

    def initialize(self, searcher: Searcher):
        self.searcher = searcher
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":[{
        #         "match_query":"match_query",
        #         "answer":"answer",
        #         "score":"score"
        #     }]
        # }

    async def post(self):
        answers = self.searcher.search(json_decode(self.request.body).get("query", ""))
        result = []
        for answer in answers:
            tmp_result = {}
            # tmp_result["query"] = answer[0]
            tmp_result["answer"] = answer[1][1]["answer"]
            tmp_result["match_query"] = answer[1][0]
            tmp_result["score"] = str(answer[3])
            result.append(copy.deepcopy(tmp_result))
        response_body = {"answer": result}
        self.write(response_body)


class VecModelHandler(RequestHandler):

    def initialize(self, vec_model: VectorizeModel):
        self.vec_model = vec_model
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":[{
        #         "query":"query",
        #         "answer":"answer",
        #         "score":"score"
        #     }]
        # }

    async def post(self):
        vec_result = self.vec_model.predict_vec_request(
            json_decode(self.request.body).get("query", "")
        )
        response_body = {"vec_result": vec_result}
        self.write(response_body)
