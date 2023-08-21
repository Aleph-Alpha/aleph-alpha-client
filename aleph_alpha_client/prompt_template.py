from liquid import Template

from aleph_alpha_client.prompt import Prompt


class PromptTemplate:
    def __init__(self, template_str: str) -> None:
        self.template = Template(template_str)

    def to_prompt(self, **kwargs) -> Prompt:
        return Prompt.from_text(self.template.render(**kwargs))
