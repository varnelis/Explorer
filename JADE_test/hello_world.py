import json

print("hello JADE world")


def func():
    return {"a": 1, "b": 2, "c": 3}


x = func()

try:
    with open("test_write_file.json", "w") as fp:
        json.dump(x, fp)
except:
    print("cannot write to file")
