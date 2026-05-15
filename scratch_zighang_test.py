import urllib.request
import re

req = urllib.request.Request('https://zighang.com/recruitment', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
uuids = set(re.findall(r'/recruitment/([0-9a-f\-]{36})', html))
print(f"Found {len(uuids)} UUIDs!")
