https://huggingface.co/models?pipeline_tag=feature-extraction&sort=trending&search=chinese

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-10b-chinese", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("BAAI/glm-10b-chinese", trust_remote_code=True)
model = model.half().cuda()

inputs = tokenizer("凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。", return_tensors="pt")
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
inputs = {key: value.cuda() for key, value in inputs.items()}
outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
print(tokenizer.decode(outputs[0].tolist()))
https://huggingface.co/THUDM/glm-10b-chinese#/




https://huggingface.co/BAAI/bge-large-zh-v1.5#/
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T



https://huggingface.co/maidalun1020/bce-embedding-base_v1#/
from BCEmbedding import RerankerModel

# your query and corresponding passages
query = 'input_query'
passages = ['passage_0', 'passage_1', ...]

# construct sentence pairs
sentence_pairs = [[query, passage] for passage in passages]

# init reranker model
model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")

# method 0: calculate scores of sentence pairs
scores = model.compute_score(sentence_pairs)

# method 1: rerank passages
rerank_results = model.rerank(query, passages)



https://huggingface.co/DMetaSoul/Dmeta-embedding-zh#/
from sentence_transformers import SentenceTransformer

texts1 = ["胡子长得太快怎么办？", "在香港哪里买手表好"]
texts2 = ["胡子长得快怎么办？", "怎样使胡子不浓密！", "香港买手表哪里好", "在杭州手机到哪里买"]

model = SentenceTransformer('DMetaSoul/Dmeta-embedding')
embs1 = model.encode(texts1, normalize_embeddings=True)
embs2 = model.encode(texts2, normalize_embeddings=True)

similarity = embs1 @ embs2.T
print(similarity)

for i in range(len(texts1)):
    scores = []
    for j in range(len(texts2)):
        scores.append([texts2[j], similarity[i][j]])
    scores = sorted(scores, key=lambda x:x[1], reverse=True)

    print(f"查询文本：{texts1[i]}")
    for text2, score in scores:
        print(f"相似文本：{text2}，打分：{score}")
    print()




https://huggingface.co/intfloat/multilingual-e5-large-instruct#/
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, '南瓜的家常做法')
]
# No need to add instruction for retrieval documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
]
input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
# => [[91.92852783203125, 67.580322265625], [70.3814468383789, 92.1330795288086]]







https://huggingface.co/jinaai/jina-embeddings-v2-base-code#/
!pip install transformers
from transformers import AutoModel
from numpy.linalg import norm

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
embeddings = model.encode(
    [
        'How do I access the index while iterating over a sequence with a for loop?',
        '# Use the built-in enumerator\nfor idx, x in enumerate(xs):\n    print(idx, x)',
    ]
)
print(cos_sim(embeddings[0], embeddings[1]))
>>> tensor([[0.7282]])



https://huggingface.co/BAAI/bge-small-en-v1.5#/
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T






https://huggingface.co/BAAI/bge-large-en-v1.5#/


https://huggingface.co/DMetaSoul/Dmeta-embedding-zh#/

https://huggingface.co/maidalun1020/bce-embedding-base_v1#/
https://huggingface.co/Frazic/udever-bloom-3b-sentence#/

https://huggingface.co/models?pipeline_tag=feature-extraction&language=zh&p=1&sort=created#/