from pydantic import BaseModel, Field, HttpUrl


class Article(BaseModel):
    title: str = Field(max_length=1000)
    text: str = Field(max_length=100_000)
    source_url: HttpUrl | None = None
