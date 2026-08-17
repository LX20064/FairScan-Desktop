import json
import sys

d = json.load(open(r"desktop\cmpre\uvdoc-infer\UVDoc_infer\inference.json"))
ops = d["program"]["regions"][0]["blocks"][0]["ops"]


def brief(o):
    attrs = []
    for a in o.get("A", []):
        if a["N"] in ("strides", "paddings", "dilation", "pads", "groups", "value", "mode", "perm"):
            v = a["AT"].get("D")
            if isinstance(v, list) and len(v) <= 6:
                attrs.append("%s=%s" % (a["N"], v))
    i = [x["%"] for x in o.get("I", [])]
    ot = []
    for x in o.get("O", []):
        if "TT" not in x:
            ot.append("")
            continue
        dims = x["TT"]["D"]
        s = []
        for d in dims:
            if isinstance(d, dict):
                v = d.get("D")
            else:
                v = d
            if v is not None:
                s.append(str(v))
        ot.append("[" + ",".join(s) + "]")
    return "%s I=%s O=%s %s" % (o["#"], i, ot, attrs)


start = int(sys.argv[1]) if len(sys.argv) > 1 else 495
end = int(sys.argv[2]) if len(sys.argv) > 2 else 519
for i in range(start, end):
    print(i, brief(ops[i]))
