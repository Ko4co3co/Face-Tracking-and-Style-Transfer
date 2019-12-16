class v4_label_parser():
    def __init__(self,file_path= './config/oid_v4_label_map.pbtxt'):
        self.file_path = file_path
    def load_label(self):
        labels = dict()
        fileptr = open(self.file_path,'r')
        pastnum = int()
        for line in fileptr.readlines():
            splits = line.replace(':','').replace('"','').split()
            try:
                if splits[0] == 'id':
                    pastnum = splits[1]
                if splits[0] == 'display_name':
                    labels[pastnum] = splits[1]
                    if len(splits) >=3:
                        labels[pastnum] += ' ' + splits[2]
                            
            except IndexError:
                pass
                
        return labels
        
if __name__ == "__main__":
    print('Test Code')
    file_path = './label_map/label_map.pbtxt'
    print(label_parsed(file_path).parser())
    
    