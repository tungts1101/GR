from collections import defaultdict

def summarize_each_record(record):
    avg_err = sum(x['error'] for x in record) / len(record)
    best_err = min(x['error'] for x in record)
    worst_err = max(x['error'] for x in record)

    avg_rt = sum(x['running time'] for x in record) / len(record)
    best_rt = min(x['running time'] for x in record)
    worst_rt = max(x['running time'] for x in record)

    return {
        'avg_err'  : avg_err,
        'best_err' : best_err,
        'worst_err': worst_err,
        'avg_rt'   : avg_rt,
        'best_rt'  : best_rt,
        'worst_rt' : worst_rt
    }

def summarize(records):
    ret = defaultdict(lambda: {})

    for key in records.keys():
        value = summarize_each_record(records[key])

        ret[key] = value

    return ret

records = {'no-dem2_r50_1.in': [{'file': 'no-dem2_r50_1.in', 'running time': 158.75639843940735, 'error': 0.1429857960488673}, {'file': 'no-dem2_r50_1.in', 'running time': 139.42898297309875, 'error': 0.1099645847843846}, {'file': 'no-dem2_r50_1.in', 'running time': 143.86133289337158, 'error': 0.027222127735276932}, {'file': 'no-dem2_r50_1.in', 'running time': 152.8459985256195, 'error': 2.028217714768068e-06}, {'file': 'no-dem2_r50_1.in', 'running time': 149.24945330619812, 'error': 0.0349007446650975}], 'uu-dem7_r50_1.in': [{'file': 'uu-dem7_r50_1.in', 'running time': 156.17433142662048, 'error': 0.18765243982959542}, {'file': 'uu-dem7_r50_1.in', 'running time': 143.3718113899231, 'error': 2.3698277209242937e-06}, {'file': 'uu-dem7_r50_1.in', 'running time': 148.70098090171814, 'error': 0.04166840514370621}, {'file': 'uu-dem7_r50_1.in', 'running time': 149.9010694026947, 'error': 0.09746799472790844}, {'file': 'uu-dem7_r50_1.in', 'running time': 153.46576237678528, 'error': 0.1540385358710503}], 'ga-dem2_r50_1.in': [{'file': 'ga-dem2_r50_1.in', 'running time': 140.7228455543518, 'error': 0.008519604076744526}, {'file': 'ga-dem2_r50_1.in', 'running time': 137.238915681839, 'error': 0.09523985300700161}, {'file': 'ga-dem2_r50_1.in', 'running time': 139.38574886322021, 'error': 0.07288915437221914}, {'file': 'ga-dem2_r50_1.in', 'running time': 152.7527039051056, 'error': 0.1923099261008464}, {'file': 'ga-dem2_r50_1.in', 'running time': 152.44138598442078, 'error': 2.1160791400812695e-06}]}

result = summarize(records)

print(result)

f = open("demo.txt", "w")
f.write("Small\n")
f.write("PSO\n")
f.write("dataset\t\t\tavg_err\t\tbest_err\tworst_err\tavg_rt\t\tbest_rt\t\tworst_rt\n")

for key in result.keys():
    res = result[key]

    f.write("{:20}\t\t{:8.3f}\t\t{:8.3}\t{:8.3f}\t{:8.3f}\t\t{:8.3f}\t\t{:8.3f}\n".format(
        key,res['avg_err'],res['best_err'],res['worst_err'],
        res['avg_rt'],res['best_rt'],res['worst_rt']
        ))
