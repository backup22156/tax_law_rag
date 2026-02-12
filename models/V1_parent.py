from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from google.colab import userdata
import re

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


try:
  collection = client.get_collection(name=collection_name)
except Exception:
  raise RuntimeError(f"Collection {collection_name} not found")

class ParentPipeline:

  def __init__(self, query, vectorstore, llm, initial_k=5, max_parent_num=10, k_preliminary=2, max_law_chars=400):
    self.query = query
    self.llm = llm
    self.initial_k = initial_k
    self.max_parent_num = max_parent_num
    self.k_preliminary = k_preliminary
    self.max_law_chars = max_law_chars
    self.basic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.initial_k})

  def find_none(self, text):
    none_case = ["", " ", None]
    return None if text in none_case else text

  def parent_where(self):

    child_docs = self.basic_retriever.invoke(self.query)

    parent_where_list = []
    seen = set()

    for child in child_docs:
      meta = child.metadata

      law_name = self.find_none(meta.get("law_name", ""))
      조문번호 = self.find_none(meta.get("조문번호", ""))
      항번호 = self.find_none(meta.get("항번호", ""))
      호번호 = self.find_none(meta.get("호번호", ""))

      if 호번호:
        where = {
          "$and": [
            {"law_name": {"$eq": law_name}}, 
            {"조문번호": {"$eq": 조문번호}}, 
            {"level": {"$eq": "호"}}, 
            {"호번호": {"$eq": 호번호}}
            ]
        }
        added = ("호", law_name, 조문번호, 호번호)
        if added not in seen:
          parent_where_list.append(where)
          seen.add(added)

      if 항번호:
        where = {
            "$and": [
                {"law_name": {"$eq": law_name}},
                {"조문번호": {"$eq": 조문번호}},
                {"level": {"$eq": "항"}},
                {"항번호": {"$eq": 항번호}},
            ]
        }
        added = ("항", law_name, 조문번호, 항번호)
        if added not in seen:
          parent_where_list.append(where)
          seen.add(added)

      if 조문번호:
        where = {
            "$and": [
                {"law_name": {"$eq": law_name}},
                {" 조문번호": {"$eq": 조문번호}},
                {"level": {"$eq": "조문내용"}}
            ]
        }
        added = ("조문", law_name, 조문번호)
        if added not in seen:
          parent_where_list.append(where)
          seen.add(added)

    return child_docs, parent_where_list

  def parent_retriever(self, parent_meta_list, collection):

    parents_list = []

    for where in parent_meta_list:
      if len(parents_list) >= self.max_parent_num:
        break

      parents = collection.get(where=where, include=["documents", "metadatas"])
      docs = parents.get("documents", [])
      metas = parents.get("metadatas", [])

      for docs, meta in zip(docs, metas):
        parents_list.append(Document(page_content=docs, metadata=meta))

    return parents_list

  def generate_answer(self):

    child_docs, parent_where_list = self.parent_where()
    parent_docs = self.parent_retriever(parent_where_list, collection)

    docs = child_docs + parent_docs

    context = "\n\n".join([f"[ID {i}]\n{doc.page_content[:500]}" for i, doc in enumerate(docs)])

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

    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
    chain = prompt | self.llm | JsonOutputParser()
    answer = chain.invoke({"question": self.query, "context": context})

    used_docs = []
    for x in answer.get("used_index", []):
      try:
        if int(x) < len(docs):
          used_docs.append(docs[int(x)])
      except (ValueError, IndexError):
        continue

    answer_text = answer.get("answer", "")

    return answer_text, used_docs, docs

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
