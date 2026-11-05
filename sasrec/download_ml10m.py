# -*- coding: utf-8 -*-
"""下载 MovieLens-10M（公开，无需登录）

为什么换数据集：
- ml-1m：3533 物品、60 万交互（rating>=4 后），每个物品平均 170 个交互
  -> 共现统计太稳，ItemCF 天然占优，序列模型学不出东西
- ml-10m：约 1 万物品、1000 万交互，数据量大 10 倍，序列模型才有发挥空间

下载地址：https://files.grouplens.org/datasets/movielens/ml-10m.zip
"""

import os
import sys
import urllib.request
import zipfile

URL = 'https://files.grouplens.org/datasets/movielens/ml-10m.zip'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'data')
ZIP_PATH = os.path.join(OUT_DIR, 'ml-10m.zip')


def download(url, path):
    """优先 requests（自带 certifi CA 链），失败再退回 ssl 上下文 + certifi"""
    try:
        import requests
        print(f'开始下载 {url}')
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = 0
        with open(path, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    print(f'\r  {total / 1e6:.1f} MB', end='', flush=True)
        print()
        return
    except Exception as e:
        print(f'requests 失败：{e}\n改用 urllib + certifi')
        import ssl
        import urllib.request
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, context=ctx) as resp, open(path, 'wb') as f:
            f.write(resp.read())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 60_000_000:
        print(f'已存在 {ZIP_PATH}（{os.path.getsize(ZIP_PATH) / 1e6:.1f} MB），跳过下载')
    else:
        download(URL, ZIP_PATH)
        print(f'下载完成 {os.path.getsize(ZIP_PATH) / 1e6:.1f} MB')

    extract_to = os.path.join(OUT_DIR, 'ml-10m')
    if not os.path.exists(os.path.join(extract_to, 'ratings.dat')):
        print('解压中...')
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(OUT_DIR)
        # 压缩包内目录名是 ml-10M100K
        src = os.path.join(OUT_DIR, 'ml-10M100K')
        if os.path.exists(src) and not os.path.exists(extract_to):
            os.rename(src, extract_to)
    rp = os.path.join(extract_to, 'ratings.dat')
    print(f'就绪：{rp}（{os.path.getsize(rp) / 1e6:.1f} MB）')


if __name__ == '__main__':
    main()
