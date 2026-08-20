from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl | None = None
    title: str | None = Field(default=None, max_length=1000)
    text: str | None = Field(default=None, max_length=100_000)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        has_url = self.url is not None
        title = (self.title or "").strip()
        text = (self.text or "").strip()
        if has_url:
            if title or text:
                raise ValueError("url cannot be combined with title or text")
            return self
        if not title:
            raise ValueError("title is required when url is absent")
        if self.text is not None and not text:
            raise ValueError("text must not be empty when supplied")
        return self


class Article(BaseModel):
    title: str = Field(max_length=1000)
    text: str = Field(max_length=100_000)
    source_url: HttpUrl | None = None
