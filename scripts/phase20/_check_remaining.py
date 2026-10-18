import json
cat = json.load(open('research/phase20_exhaustive/experiment_catalogue_96.json', encoding='utf-8'))
ns = [x for x in cat['experiments'] if x['status'] == 'NOT_STARTED']
for x in ns:
    print(f'{x["experiment_number"]:3d} {x["research_family"]:5s} {x["priority"]} {x["canonical_name"][:50]:50s}')