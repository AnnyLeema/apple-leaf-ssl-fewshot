import json,re,sys
def parse(path):
    rows=[];cur=None;e4=[];
    for l in open(path):
        l=l.strip()
        if l.startswith('{"acc"'):
            r=json.loads(l);rows.append(r);cur=(r['seed'],r['frac']);e4=[]
        m=re.match(r"(E4_\w+) (\{.*\})",l)
        if m and cur:
            d=eval(m.group(2));e4.append(m.group(1))
            rows.append(dict(name=m.group(1),seed=cur[0],frac=cur[1],fewshot={k.replace('_K5','')+'_K5':dict(acc=v) for k,v in d.items()}))
    return rows
a=parse('/mnt/user-data/uploads/4b0cc82b-attachment.txt');b=parse('/mnt/user-data/uploads/220a1771-attachment.txt')
json.dump(a,open('old_seeds012.json','w'));json.dump(b,open('new_seeds34567.json','w'))
print(len(a),len(b))
