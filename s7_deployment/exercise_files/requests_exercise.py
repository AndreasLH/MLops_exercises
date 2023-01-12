import requests
personal_acces_token = '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'
headers = {'Authorization': personal_acces_token}

response = requests.get('https://api.github.com/this-api-should-not-exist', headers=headers)
print(response.status_code)

response = requests.get('https://api.github.com', headers=headers)
print(response.status_code)


response = requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops", headers=headers)
response.json()

response = requests.get(
   'https://api.github.com/search/repositories',
   params={'q': 'requests+language:python'}, headers=headers
)



# response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')
# with open(r'img.png','wb') as f:
#    f.write(response.content)

pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)
a = 2