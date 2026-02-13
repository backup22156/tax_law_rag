from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import json
import re
from langchain_classic.retrievers import EnsembleRetriever

from google.colab import userdata

llm = ChatOpenAI(
    model = "gpt-4.1-nano",
    temperature = 0,
    max_tokens = 2000,
    openai_api_key = userdata.get('OPENAI_API_KEY')
)

db_path = "/content/drive/MyDrive/rag/chromadb_backup"
client = chromadb.PersistentClient(path=db_path)
collection_name = "tax_law"

try:
  collection = client.get_collection(name=collection_name)
except Exception:
  raise RuntimeError(f"Collection {collection_name} not found")

embedding_model = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key = userdata.get("OPENAI_API_KEY"))

vectorstore = Chroma(client = client, collection_name=collection_name, embedding_function=embeddings)


class LLMReranker:

  def __init__(self, llm, top_k=7):
    self.llm = llm
    self.top_k = top_k
    self.prompt_reranking = """
    너는 세법 전문가야. 질문의 의도에 가장 적절한 답변을 제공하는 상위 {top_k}개의 문서들을 선별해.

    - 질문에 포함된 수치가 단순히 언급된 것이 아니라, 질문이 요구하는 '논리적 정의', '조건', '한도', '계산근거'로서 설명되는 문서를 우선시해.
    - 질문과 무관한 법령은 단어가 겹쳐도 낮은 점수를 줘.
    - 문서 간 상대적 우위를 비교하여 상위 {top_k}개의 문서들에 대해 0~100점 사이로 차등 점수를 줘.

    [질문]
    {question}

    [문맥]
    {context}

    [출력 형식 - JSON]
    {{
      "scores": [
        {{"id": 0, "score": 93, "reason": "법적 조건인 연간 소득금액 100만원에서 소득금액의 정의를 구체적으로 명시함"}},
        {{"id": 3, "score": 65, "reason": "연간 소득금액 100만원의 합산 범위를 구체적으로 명시함"}},
        ...
      ]
    }}
    """
    self.prompt = PromptTemplate(input_variables=["question", "context", "top_k"], template=self.prompt_reranking)

  def rerank(self, docs, query):

    docs_matching = {str(i): doc for i, doc in enumerate(docs)}

    docs_formatted = "\n".join([
        f"[ID {i}] ([출처] law name: {doc.metadata.get('law_name')}, 조문제목: {doc.metadata.get("조문제목")}) {doc.page_content[:500]}"
        for i, doc in enumerate(docs)])
    
    chain = self.prompt | self.llm | JsonOutputParser()

    rerank_result = chain.invoke({"question": query, "context": docs_formatted, "top_k": self.top_k})
    scores = rerank_result.get("scores", [])

    scored_docs = []
    for x in scores:
      id = str(x.get("id"))
      score = int(x.get("score", 0))
      if id in docs_matching:
        scored_docs.append((docs_matching[id], score))

    scored_docs.sort(key=lambda x: x[1], reverse = True)

    reraked_docs = [doc for doc, _ in scored_docs[:self.top_k]]

    return reraked_docs

class AdvancedPipeline:

  def __init__(self, query, llm, vectorstore, initial_k=15, final_k=20, top_k=7):
    self.initial_k = initial_k
    self.final_k = final_k
    self.top_k = top_k
    self.query = query
    self.llm = llm
    self.vectorstore = vectorstore

    all_data = vectorstore.get()
    all_docs = [
        Document(page_content= content, metadata=metadata)
        for content, metadata in zip(all_data['documents'], all_data['metadatas'])
    ]

    self.bm25_retriever = BM25Retriever.from_documents(all_docs, search_kwargs={"k": self.initial_k})
    self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": self.initial_k})
    self.ensemble_retriever = EnsembleRetriever(retrievers = [self.bm25_retriever, self.vector_retriever], weights = [0.3, 0.7])

    self.reranker = LLMReranker(llm, self.top_k)


  def generate_queries(self):
    prompt_multi_query = """
    사용자의 질문을 분석해서 다음 3가지 관점의 세법 전문가용 검색어로 변환해줘
    [요건 관점] 질문의 상황과 관련된 법령 적용의 직접적인 요건 (ex. 부양가족 공제 소득요건)
    [정의 관점] 질문에 포함된 핵심 용어의 법적 정의 (ex. 소득세법상 소득금액의 정의 및 범위)
    [계산 관점] 적용되는 산식, 계산 방법

    실제 법령과 집행기준, 판례에 쓰이는 전문 용어를 사용해.

    [질문]
    {question}

    [출력형식]
    검색어를 줄바꿈만으로 구분
    """ 

    prompt = PromptTemplate(input_variables=["question"], template=prompt_multi_query)
    chain = prompt | self.llm | StrOutputParser()
    multi_query = chain.invoke({"question": self.query})
    multi_query = multi_query.split("\n")
    multi_query.append(self.query)
    return multi_query

  def retrieve(self):
    multi_query = self.generate_queries()

    k_const = 60

    docs_score = {}
    docs_list = {}
    for query in multi_query:
      docs = self.ensemble_retriever.invoke(query)
      for rank, doc in enumerate(docs):
        content = doc.page_content

        score = 1.0/(k_const + rank)
        if content in docs_score:
          docs_score[content] += score
        else:
          docs_score[content] = score
          docs_list[content] = doc

    sorted_docs = sorted(docs_score.items(), key=lambda x: x[1], reverse=True)
    sorted_list = [docs_list[content] for content, _ in sorted_docs[:self.final_k]]

    reranked_docs = self.reranker.rerank(sorted_list, self.query)

    return reranked_docs

  def generate_answer(self):
    reranked_docs = self.retrieve()

    context = "\n\n".join([
        f"[ID {i}] ([출처] law name: {doc.metadata.get('law_name')}, 조문제목: {doc.metadata.get('조문제목')})\n{doc.page_content[:500]}"
        for i, doc in enumerate(reranked_docs)
    ])

    # print(context)

    prompt_final = """
    너는 세법 전문가야. 아래의 문맥을 근거로 질문에 답변해.

    [규칙]
    1. 첫 줄에 한 문장 결론을 제시하되, 다음 중 하나로 시작
    - 결론(확정): ...
    - 결론(조건부): ...
    2. 문맥에서 지지되는 내용만 '문맥 기반 결론'에 쓸 것
    3. 질문에 대해 가능한 답을 하되, 불확실하면 조건부로 분기
    4. '문맥 기반 결론'에 포함되는 모든 문장 끝에 반드시 해당 문서 ID인 [[n]]를 기입할 것.
    5. used_index는 '문맥 기반 결론'에 등장한 [[n]]만 그대로 추출하여 리스트로 만들 것.
    6. 예외/단서("다만", "~의 경우", 부정요건)가 문맥에 있으면 반드시 반영하고, 예외가 원칙보다 우선함을 고려할 것
    7. 문맥에 직접적인 문구가 없더라도, 아래 조건을 모두 만족하면 '조건부 판단'에서만 제한적으로 추론 가능
    - 문맥에 관련 조문/정의/요건의 일부가 존재하고[[n]]으로 인용 가능
    - 추론을 "만약 ~라면" 형태로 조건을 붙여 분기
    - 추론 문장 끝에 [[n]]을 억지로 붙이지 말 것
    8. "제공된 자료에서 관련 내용을 찾을 수 없습니다"는 아래의 경우에만 사용:
    - 문맥에 질문의 핵심 대상/행위/제도가 전혀 없고
    - 관련 조문/정의/요건/예외를 단 하나도 찾지 못한 경우
    9. 출력형식은 반드시 JSON 형식 유지

    [작성 순서]
    1단계: 문맥에서 관련 조항(정의/요건/원칙/예외)를 모두 찾아 핵심 문장 추출
    2단계: 예외/단서가 원칙보다 우선하는지 판단
    3단계: (확정 가능하면) 확정 결론, 아니면 조건부 결론+추가 확인 질문

    [문맥]
    {context}

    [질문]
    {question}

    [출력 형식-JSON]
    {{
      "answer": "두괄식 답변 및 논리적 근거 (근거 문장 끝에 [[n]])",
      "used_index": ["0", "3"]
    }}
    """

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_final)
    chain = prompt | self.llm | JsonOutputParser()
    answer = chain.invoke({"context": context, "question": self.query})

    used_docs = []
    for x in answer.get("used_index", []):
      try:
        if int(x) < len(reranked_docs):
          used_docs.append(reranked_docs[int(x)])
      except (ValueError, IndexError, TypeError):
        continue

    answer_text = answer.get("answer", "답변을 생성할 수 없습니다.")

    return answer_text, used_docs, reranked_docs


class Reference:

  def strip_dot(self, text):
    if not text:
      return ""
    return text.rstrip(".")

  def clean_content(self, text):
    if not text:
      return text

    text = text.strip()

    text = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]+\s*", "", text)
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"^[가-힣]\.\s*", "", text)

    return text.strip()

  def make_reference(self, doc, max_law_chars=300):
    출처 = ""
    meta = doc.metadata
    law_name = meta.get("law_name")
    출처 += law_name
    조문번호 = meta.get("조문번호", "")
    출처 += f" {조문번호}조" if 조문번호 else ""
    항번호 = meta.get("항번호", "")
    출처 += f" {self.strip_dot(항번호)}항" if 항번호 else ""
    호번호 = meta.get("호번호", "")
    출처 += f" {self.strip_dot(호번호)}호" if 호번호 else ""
    목번호 = meta.get("목번호", "")
    출처 += f" {self.strip_dot(목번호)}목" if 목번호 else ""

    content = self.clean_content(doc.page_content)
    if len(content) > max_law_chars:
      content = content[:max_law_chars] + "..."

    return 출처, content

  def generate_reference(self, pipeline):
    answer_text, used_docs, _ = pipeline.generate_answer()

    reference_list= []
    for doc in used_docs:
      출처, content = self.make_reference(doc)
      reference_list.append(f"{출처}: {content}")

    final_answer = answer_text + "\n\n[근거 법령 및 본문]" + "\n" .join(reference_list)

    return final_answer
  











