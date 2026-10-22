import sys,collections,re
from gguf import GGUFReader
r=GGUFReader(sys.argv[1])
c=collections.Counter()
for t in r.tensors:
    c[(re.sub(r"\.\d+\.",".N.",t.name),t.tensor_type.name)]+=1
for (n,ty),v in sorted(c.items()): print(v,n,ty)
