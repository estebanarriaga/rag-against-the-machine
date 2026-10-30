import dspy
from index_module import Indexer

def open_file(path: str):
    try:
        file_stream = open(path, mode='r')
    except IOError as e:
        print(e)
    return file_stream

def main():
    lm = dspy.LM(
    model="openai/qwen3:0.6b",
    api_base="http://localhost:11434/v1",
    api_key="EMPTY",
    temperature=0.3,
    max_tokens=2048,
)
    try:
        dspy.configure(lm=lm)
        # predict = dspy.Predict("question -> answer")
        # result = predict(question="What is a binary heap?")
        indexer = Indexer("./vllm-0.10.1", max_chunk_size=2000)
        indexer.create_index()
        # file = open_file("vllm-0.10.1/README.md")
        # content = file.read()
        # print(content)
        
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
