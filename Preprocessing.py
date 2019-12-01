import os
import json
"""
Process the raw data into: 

1. one file that contains all the sentence
2. one file contains the correponded relation ship
[{origin:(record_num, line), 
 line: 20, c1:("concept 1", 0, 1), 
 r:"TrCP", c2:("concept 2", 7, 8)
},...,]
3. one file contains the segment of the new sentence:

....

Eventally, get a tensor:
each word is a nw embedding

"""

class Process:
    def __init__(self, path):
        self.name = path.split("/")[-1].split(".")[:-1][0]
        with open(path) as f:
            j = []
            for line in f:
                j.append(self.lineprocess(line))
        self.data = j
        """ if j != {}:
            print(self.data) """
    def lineprocess(self):
        return {}
    
    

    


class Relation(Process):
    def __init__(self, path):
        super().__init__(path)
        self.txt = None
    def lineprocess(self, line):
        [c1, r, c2] = [t.split("=")[1] for t in line.split("||")]
        cobj = lambda c: {'value': " ".join(c.split()[:-2]).replace('"',''), 
                          'span': [int(m.split(":")[1]) for m in c.split()[-2:]],
                        }
        relobj = {'value': "".join([i for i in r][1:-1])}
        c1obj = cobj(c1)
        c2obj = cobj(c2)
        # line num is the same?
        assert int(c1.split()[-1].split(":")[0]) == int(c2.split()[-1].split(":")[0])
        origin = {'line': int(c1.split()[-1].split(":")[0]), 'name': self.name}
        obj = {'c1': c1obj, 'c2':c2obj, 'r': relobj, 'origin': origin}
        return obj
    
    def get_txt(self, txtobj):
        for i in self.data:
            # print(self.name)
            line_index = i['origin']['line'] - 1
            # print(txtobj.data[line_index])
            i.update(txtobj.data[line_index])
    
    
            

class Text(Process):
    def lineprocess(self, line):
        return {'txt': line.rstrip().replace('"','').replace("&", "and")}
    

paths = [
            "/home/compmed/CompMed-RE/data/concept_assertion_relation_training_data/beth/",
            "/home/compmed/CompMed-RE/data/concept_assertion_relation_training_data/partners/"
        ]


old_path = os.getcwd()
total_train_json = []
for path in paths:
    os.chdir(path)
    filelength = lambda x: len([f for f in os.listdir(x) if not f.startswith('.')]) 

    assert filelength("./txt/") == filelength("./rel/")
    record_num = filelength("./txt/")
    # relation:
    filelist_rel = [".".join(f.split(".")[:-1]) for f in os.listdir(path + 'rel/') if not f.startswith('.')]
    all_instances = []
    for i in filelist_rel:
        p_file_rel = path + 'rel/' + i + '.rel'
        relobj = Relation(p_file_rel)
        p_file_txt = path + 'txt/' + i + '.txt'
        txtobj = Text(p_file_txt)
        relobj.get_txt(txtobj)
        all_instances += relobj.data
    total_train_json += all_instances
print(total_train_json[0:3])
os.chdir(old_path)


# dump file as json
with open("processed_train.json", 'w') as f:
    json.dump(total_train_json, f)

        

        

    
        





#path = "/home/compmed/CompMed-RE/data/concept_assertion_relation_training_data/partners/rel/402389409_WGH.rel"
#m = Relation(record_num, path)
    