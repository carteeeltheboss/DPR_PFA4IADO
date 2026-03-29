import re

with open('beamer.tex', 'r') as f:
    content = f.read()

def repl(m):
    # remove all double newlines (blank lines)
    inner = m.group(0)
    inner_fixed = re.sub(r'\n[ \t]*\n', '\n', inner)
    return inner_fixed

content = re.sub(r'\\tikzsolutionfigure\{%.*?\}\{.*?\}', repl, content, flags=re.DOTALL)

with open('beamer.tex', 'w') as f:
    f.write(content)
