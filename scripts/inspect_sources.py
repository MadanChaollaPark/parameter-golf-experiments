import json, os, urllib.request, pathlib, datetime
out=pathlib.Path('evidence'); out.mkdir(exist_ok=True)
def get(url,auth=False):
    h={'User-Agent':'parameter-golf-independent-research'}
    if auth: h['Authorization']='Bearer '+os.environ['GH_TOKEN']
    with urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=60) as r:return json.load(r)
u=get('https://api.github.com/user',True); print('GitHub authenticated:',u['login'])
repo=get('https://api.github.com/repos/openai/parameter-golf/commits/main'); print('upstream',repo['sha'])
for n in [2135,2014,2127,1017]:
    for suffix in ['', '/comments?per_page=100']:
        data=get(f'https://api.github.com/repos/openai/parameter-golf/issues/{n}{suffix}')
        (out/f'github-{n}{"-comments" if suffix else ""}.json').write_text(json.dumps(data,indent=2))
    issue=json.loads((out/f'github-{n}.json').read_text()); print(n,issue['title'],issue['state'])
    comments=json.loads((out/f'github-{n}-comments.json').read_text())
    for c in comments:
        if c.get('author_association') in ['OWNER','MEMBER','COLLABORATOR']:print(n,c['user']['login'],c['author_association'],c['body'][:15000])
(out/'upstream.json').write_text(json.dumps({'sha':repo['sha'],'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
url="https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20'belgiumcentral'%20and%20armSkuName%20eq%20'Standard_D4s_v4'%20and%20priceType%20eq%20'Consumption'"
a=get(url); (out/'azure-price.json').write_text(json.dumps(a,indent=2));print('Azure:',[(i['productName'],i['skuName'],i['retailPrice'],i['unitOfMeasure']) for i in a['Items']])
