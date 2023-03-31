import json

af = open("data/yelp2018/yelp_new_review34.json", "r").readlines()

begin = False
count = 0

with open("data/yelp2018/yelp_new_review35.json", "w") as f:
    for l in af:
        dic = json.loads(l)
        count += 1
        if dic['user_remap_id'] == 24401 and dic["item_remap_id"] == 16371:
            begin = True
            print(count)
            continue
        if begin:
            json.dump({'user_remap_id': dic['user_remap_id'], 'item_remap_id': dic['item_remap_id'], 'review': dic['review']}, f)
            f.write('\n')

