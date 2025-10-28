import time
import json
import pandas as pd
from tqdm import tqdm
import execjs
import argparse

# help_fun 针对于value是list的还应该进一步考虑
def dict_value(data, res, front=""):
    for k, v in data.items():
        if isinstance(v, dict):
            dict_value(v, res, front=f"{k}_")
        else:
            res.append(f"{front+k}={v}")


def list2set(k, v, max=10):
    v = set(v)
    v = list(v)
    v = v[:max]
    v = [str(_) for _ in v]
    # v = ["@"+k+"/"+str(_) for _ in v]
    # v[0] = v[0].split("/")[1]
    return "/".join(v)


def process_list_value(k, v):
    # 如果元素都是int类型则需要先转str再concat
    if len(v) == 0:
        return ''
    if isinstance(v[0], int):
        if k in ["sample_offsets", "chunk_offsets", "sample_sizes", "first_chunk", "samples_per_chunk", "chunk_offsets", "sample_description_index",
                 "sample_has_redundancy", "sample_is_depended_on", "sample_depends_on", "is_leading", "sample_numbers",
                 "sample_deltas", "sample_counts"]:
            # return str(len(v))
            return list2set(k, v)
        return "_".join('%s'%item for item in v)
    if isinstance(v[0], str):
        if k in ["sample_offsets", "chunk_offsets", "sample_sizes", "first_chunk", "samples_per_chunk", "chunk_offsets", "sample_description_index",
                 "sample_has_redundancy", "sample_is_depended_on", "sample_depends_on", "is_leading", "sample_numbers",
                 "sample_deltas", "sample_counts"]:
            # return str(len(v))
            return list2set(k, v)
        return "_".join(v)
    if isinstance(v[0], dict):
        # 待定
        res = []
        dict_value(v[0], res)
        return "&".join(res)
    if isinstance(v[0], list):
        data_str = '/'.join([f"data{i + 1}=" + '&'.join([f"{k}:{v}" for k, v in d[0]['data'].items()]) for i, d in enumerate(v)])
        # [
        #     [{'data': {'0': 64, '1': 1}}],
        #     [{'data': {'0': 63, '1': 1}}],
        #     [{'data': {'0': 62, '1': 1}}]
        #  ]
        # data1=0:64&1:1/data2=0:63&1:1/data3=0:62&1:1
        # print(data_str)
        return data_str


def help_fun_v2(front, box, result, idx):
    """
    :param front:   前缀
    :param box:     当前box
    :param result:  最终存的结果的list
    :return:        None
    """
    # 存在还有boxes的情况 递归向下
    if box.get("boxes") is not None:
        for index, b in enumerate(box["boxes"]):
            help_fun_v2(f'{front+box["type"]}-{idx}/', b, result, index+1)
    elif box.get("entries") is not None:
        for index, b in enumerate(box["entries"]):
            help_fun_v2(f'{front+box["type"]}-{idx}/', b, result, index+1)
    else:
        # 前缀 缺省为0
        try:
            front += f'{box["type"]}-{idx}'
        except:
            front += f"None-{idx}"
        # 当前box下的所有节点及其对应的值
        for k, v in box.items():
            if k == "type":
                continue
            ### v2不需要存key ###
            # 存key /@key
            # result.append(front+"/@"+k)
            ### v2不需要存key ###
            # 存value /@key/value
            # value存在多种情况 int或str或list
            if isinstance(v, int):
                result.append(front+"/@"+k+":"+str(v))
            elif isinstance(v, str):
                result.append(front+"/@"+k+":"+v)
            elif isinstance(v, list):
                try:
                    # result.append(front+"/@"+k+":"+str(v))
                    # print(v)
                    result.append(front+"/@"+k+":"+process_list_value(k, v))
                except Exception as e:
                    print(f"{v} err, Message: {e}")
                    # pass


def help_fun_v3(front, box, result):
    """
    :param front:   前缀
    :param box:     当前box
    :param result:  最终存的结果的list
    :return:        None
    """
    # 区别：ftyp 而不是 ftyp-1
    # 存在还有boxes的情况 递归向下
    if box.get("boxes") is not None:
        for index, b in enumerate(box["boxes"]):
            help_fun_v3(f'{front+box["type"]}/', b, result)
    elif box.get("entries") is not None:
        for index, b in enumerate(box["entries"]):
            help_fun_v3(f'{front+box["type"]}/', b, result)
    else:
        # 前缀 缺省为0
        try:
            front += f'{box["type"]}'
        except:
            front += f"None"
        # 当前box下的所有节点及其对应的值
        for k, v in box.items():
            if k == "type":
                continue
            ### v2不需要存key ###
            # 存key /@key
            # result.append(front+"/@"+k)
            ### v2不需要存key ###
            # 存value /@key/value
            # value存在多种情况 int或str或list
            if isinstance(v, int):
                result.append(front+"/@"+k+":"+str(v))
            elif isinstance(v, str):
                result.append(front+"/@"+k+":"+v)
            elif isinstance(v, list):
                try:
                    # result.append(front+"/@"+k+":"+str(v))
                    # print(v)
                    result.append(front+"/@"+k+":"+process_list_value(k, v))
                except Exception as e:
                    print(f"{v} err, Message: {e}")
                    # pass


def help_fun_v4(front, box, result, idx):
    """
    :param front:   前缀
    :param box:     当前box
    :param result:  最终存的结果的list
    :return:        None
    """
    # ftyp/1 而不是 ftyp-1
    # 存在还有boxes的情况 递归向下
    if box.get("boxes") is not None:
        for index, b in enumerate(box["boxes"]):
            help_fun_v4(f'{front+box["type"]}/{idx}/', b, result, index+1)
            # help_fun_v4(f'{front+box["type"]}-{idx}/', b, result, index+1)
    elif box.get("entries") is not None:
        for index, b in enumerate(box["entries"]):
            help_fun_v4(f'{front+box["type"]}/{idx}/', b, result, index+1)
            # help_fun_v4(f'{front+box["type"]}-{idx}/', b, result, index+1)
    else:
        # 前缀 缺省为0
        try:
            front += f'{box["type"]}/{idx}'
            # front += f'{box["type"]}-{idx}'
        except:
            front += f"None/{idx}"
            # front += f"None-{idx}"
        # 当前box下的所有节点及其对应的值
        for k, v in box.items():
            if k == "type":
                continue
            ### v2不需要存key ###
            # 存key /@key
            # result.append(front+"/@"+k)
            ### v2不需要存key ###
            # 存value /@key/value
            # value存在多种情况 int或str或list
            if isinstance(v, int):
                result.append(front+"/@"+k+":"+str(v))
            elif isinstance(v, str):
                result.append(front+"/@"+k+":"+v)
            elif isinstance(v, list):
                try:
                    # result.append(front+"/@"+k+":"+str(v))
                    # print(v)
                    result.append(front+"/@"+k+":"+process_list_value(k, v))
                except Exception as e:
                    print(f"{v} err, Message: {e}")
                    # pass



def get_video_info(video_path):
    with open('extract_video_info.js', 'r', encoding='utf-8') as f:
        js_code = f.read()

    node = execjs.get()
    # print(node.name)
    ctx = node.compile(js_code)
    # print("JS compiled successfully")
    max_retries = 10
    retry_count = 0
    result = ctx.call("getMsg", video_path, '-1', 0)
    while result == 'isReadyFalse':
        time.sleep(1)  # 注意：execjs 是同步的！下面说明问题
        result = ctx.call("getMsg", video_path, '-1', 0)
        retry_count += 1
        if retry_count >= max_retries:
            return None

    keys_default = ['type', 'size', 'hdr_size', 'start', 'major_brand', 'minor_version', 'compatible_brands', 'boxes', 'subBoxNames', 'traks', 'psshs', 'mvhd', 'free', 'udta', 'ICAT']
    re = []
    _keys = ctx.call("getMsg", video_path, '-1', 0)
    for num, keys in _keys.items():
        re1 = {}
        for key in keys:
            if key not in keys_default:
                continue
            if key == 'boxes' or key == 'traks':
                # 获取完整分块（自动递归处理嵌套）
                boxes = []
                index = 0
                while True:
                    result = ctx.call("getMsg", video_path, key + '_' + str(index), num)
                    if result == {}:
                        break
                    key_b = result['key']
                    value_b = result['value']
                    boxes.append(value_b)
                    index += 1
                re1[key] = boxes
            else:
                result = ctx.call("getMsg", video_path, key, num)
                re1[key] = result
        re.append(re1)
    return re


def v2_save_all_data_to_json(csv_path="filename/path_ECOMMERCE.csv", save_path="vision.json"):
    print(csv_path)
    print(save_path)
    node = execjs.get()
    print(node.name)
    df = pd.read_csv(csv_path)
    video_path_list = df["filename"].to_list()
    label_list = df["label"].to_list()
    data = []
    # ctx = node.compile(open('parse.js', 'r', encoding='utf-8').read())
    for vidx, video_path in enumerate(tqdm(video_path_list)):
        video_path = video_path_list[vidx]
        label = label_list[vidx]
        box_dict = get_video_info(video_path)
        # box_dict = ctx.call("getMsg", video_path)
        result = []
        for idx, box in enumerate(box_dict):
            help_fun_v4("", box, result, idx+1)
        result = [s.replace(":", "/").replace("&", "/").replace("=", "/").replace(" ", "") for s in result]
        result = [s[:-1] if s[-1] == "/" else s for s in result ]
        data.append({
            "filename": video_path,
            "label": label,
            "data": result
        })
    f = open(save_path, "w", encoding="utf-8")
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save CSV data to JSON file.")
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to the input CSV file.'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path to save the output JSON file.'
    )

    args = parser.parse_args()

    v2_save_all_data_to_json(
        csv_path=args.csv_path,
        save_path=args.save_path
    )