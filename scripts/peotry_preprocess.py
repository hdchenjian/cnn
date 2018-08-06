# -*- coding: utf-8 -*-
import traceback

poetry_file ="poetry.txt"
poetry_file ="poetry_small.txt"

data = open("poetry_small_.txt", "w")

count = 0
with open(poetry_file,"r", encoding ="utf-8") as f:
    for line in f:
        count += 1
        #print(count)
        try:
            title, content=line.strip().split(":")
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if '（' in title or '）' in title:
                continue
            if len(content) <5 or len(content) >79 or len(title) >10:
                continue
            title = title.replace('。', '-')
            #print(content)
            #print(title)
            #print('content')

            data.write(title + ':' + content + '\n')
            #print('content')
        except Exception as e:
            traceback.print_exc()
            break;
data.close()
