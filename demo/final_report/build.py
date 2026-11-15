"""
Builds the end-term report as one self-contained HTML file (all figures embedded).

    python demo/final_report/build.py [output.html]

Default output: ~/Downloads/endterm_report/DistributedRL_Archit.html
"""
import base64
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        '~/Downloads/endterm_report/DistributedRL_Archit.html')
    html = open(os.path.join(HERE, 'report.html')).read()

    def embed(match):
        path = os.path.join(HERE, match.group(1))
        data = base64.b64encode(open(path, 'rb').read()).decode()
        return 'src="data:image/png;base64,%s"' % data

    html, n = re.subn(r'src="(figures/[^"]+\.png)"', embed, html)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(html)
    print('wrote %s (%d figures, %.1f MB)' % (out, n, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
