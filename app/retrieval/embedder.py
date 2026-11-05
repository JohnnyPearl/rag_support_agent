from openai import AsyncOpenAI, OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str):
        self.model = model
        self._client = OpenAI(api_key=api_key)
        self._aclient = AsyncOpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in response.data)
        return out

    async def aembed_query(self, text: str) -> list[float]:
        response = await self._aclient.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding
