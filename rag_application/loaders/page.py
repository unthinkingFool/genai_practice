from langchain_community.document_loaders import WebBaseLoader

url="http://support.apple.com/en-us/125405"

loader = WebBaseLoader(url)
docs = loader.load()
print(docs[0].page_content)