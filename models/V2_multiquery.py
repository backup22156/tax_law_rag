from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from google.colab import userdata
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import json
import re


llm = ChatOpenAI(
  model = "gpt-4.1-nano",
  temperature = 0,
  max_tokens = 1000,
  openai_api_key = userdata.get("OPENAI_API_KEY")
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


class LLMRerank:

  def __init__(self, llm):
    self.llm = llm

    self.prompt_rerank = """
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

  def rerank(self, query, docs, final_k):
    
    docs_matching = {str(i): doc for i, doc in enumerate(docs)}
    docs_formatted = "\n".join([f"[문서 ID: {i}] {doc.page_content[:500]}..." for i, doc in docs_matching.items()])
    
    prompt = PromptTemplate(input_variables=["question", "context", "top_k"], template=self.prompt_rerank)
    chain = prompt | self.llm | JsonOutputParser()

    try:
      result = chain.invoke({"question": query, "context": docs_formatted, "top_k": final_k})
      scores = result.get("scores", [])

      scored_docs = []
      for x in scores:
        id = str(x.get("id"))
        score = int(x.get("score", 0))

        if id in docs_matching:
          scored_docs.append((docs_matching[id], score))

      scored_docs.sort(key=lambda x: x[1], reverse=True)
      reranked_docs = [doc for doc, _ in scored_docs[:final_k]]

      return reranked_docs

    except Exception as e:
      print(f"reranking 오류: {e}")
      return docs[:final_k]


class MultiQueryPipeline:

  def __init__(self, query, llm, vectorstore, initial_k=15, final_k=7):
    self.query = query
    self.llm = llm
    self.initial_k = initial_k
    self.final_k = final_k

    self.basic_retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k": initial_k})

    self.multi_query_prompt = """
    질문을 세법 전문 용어를 사용해 3개의 서로 다른 query로 재작성해.
    각 쿼리는 반드시 줄바꿈으로 구분해.

    질문: {question}
    """

    self.prompt = PromptTemplate(input_variables=["question"], template = self.multi_query_prompt)

    self.multiquery_retriever = MultiQueryRetriever.from_llm(
      retriever = self.basic_retriever,
      llm = self.llm,
      prompt = self.prompt,
      include_original = True
    )

  def reranking(self):
    docs = self.multiquery_retriever.invoke(self.query)

    reranker = LLMRerank(self.llm)
    reranked_docs = reranker.rerank(self.query, docs, self.final_k)

    context = "\n\n".join([f"[ID {i}]\n{doc.page_content[:500]}" for i, doc in enumerate(reranked_docs)])

    return context, reranked_docs

  def generate_answer(self):
    context, reranked_docs = self.reranking()

    prompt_template = """
    너는 세법 전문가야. 아래의 문맥을 근거로 질문에 답변해.

    [규칙]
    1. 질문에 대한 답은 한 문장으로 먼저 제시 (ex. "납부할 세액은 1,000,000원입니다." "해당 항목은 익금산입 대상입니다.")
    2. 답변 근거가 된 문장 끝에 반드시 해당 문서 ID인 [[번호]]를 기입할 것.
    3. used_index는 답변 근거에 등장한 [[번호]]만 그대로 추출하여 리스트로 만들 것.
    4. 답변시 법령의 예외 조항이나 단서 조항(다만, ~의 경우 등)이 있는지 확인해서 반영할 것.
    5. 문맥이 질문에 대한 답변 근거를 제공하지 못하면, "제공된 자료에서 관련 내용을 찾을 수 없습니다"로 답하고 used_index는 빈 리스트 [] 제공.

    [문맥]
    {context}

    [질문]
    {question}

    [출력 형식-JSON]
    {{
      "answer": "두괄식 답변 및 논리적 근거 (각 근거 문장 끝에 [[n]])",
      "used_index": ["0", "3"]
    }}
    """

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    chain = prompt | self.llm | JsonOutputParser()
    answer = chain.invoke({"context": context, "question": self.query})

    used_docs = []
    for x in answer.get("used_index", []):
      try:
        if int(x) < len(reranked_docs):
          used_docs.append(reranked_docs[int(x)])
      except (ValueError, IndexError):
        continue

    answer_text = answer.get("answer", "")

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


