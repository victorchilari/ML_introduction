# Work With Txt

def intoFile(name, content, sec_content=""):
    # round float
    content = content if not isinstance(content, float) else round(content, 2)
    sec_content = sec_content if not isinstance(sec_content, float) else round(sec_content, 2)

    f = open(name + ".txt", "w")
    if sec_content == "":
        text = content
    else:
        text = str(content) + " " + str(sec_content)
    f.write(str(text))
    f.close()


def printFile(name):
    f = open(name + ".txt", "r")
    print(name + "\n", f.read())