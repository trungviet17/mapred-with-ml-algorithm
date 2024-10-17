from mrjob.job import MRJob 
from mrjob.step import MRStep
from collections import defaultdict
import math 


class MRNaiveBayes(MRJob):
    """
    Chia thành 2 steps : 
        1. Tính xác suất của sự kiện -> tương tự wordcount (khác key)
        2. Từ xác suất đã tính, nhận đầu vào là input mới chưa có nhãn 
            -> tính xác xuất từng trường xảy ra của từng trường hợp -> so sánh -> đưa ra nhãn 
    """
    def configure_args(self): 
        """
        thêm đường dẫn tới file centroids vào cmd 
        """
        super(MRNaiveBayes, self).configure_args()
        self.add_file_arg('--input_path', help = 'input file path ')

    
    def load_inut(self): 
        
        inputs = []
        with open(self.options.input_path, 'r') as file: 
            for line in file: 
                line = line.strip().split(',')
                inputs.append(
                    (
                        'outlook', line[0], 
                        'temp', line[1], 
                        'humidity', line[2], 
                        'wind', line[3]
                    )
                )
        return inputs

    def mapper_init(self): 
        self.inputs = self.load_inut()



    def mapper_train(self, _, line):
        line = line.strip().split(',') 
        target = line[-1] 
        label = ['outlook', 'temp', 'humidity', 'wind']
        for i in range(len(line) -1) : 
            yield (label[i], line[i], target) , 1
        yield ('play', target), 1


    def reducer_train(self, key, value): 
        yield key, sum(value)

    
    def mapper_infer(self, key, value):
        
        for i in range(len(self.inputs)): 
            if key[0] == 'outlook': 
                if key[1] == self.inputs[i][1]: 
                    yield self.inputs[i], (key, value)
            if key[0] == 'temp': 
                if key[1] == self.inputs[i][3]: 
                    yield self.inputs[i], (key, value)
            if key[0] == 'humidity': 
                if key[1] == self.inputs[i][5]: 
                    yield self.inputs[i], (key, value)
            if key[0] == 'wind': 
                if key[1] == self.inputs[i][7]: 
                    yield self.inputs[i], (key, value)
            if key[0] == 'play': 
                yield self.inputs[i], (key, value)

    def reducer_infer(self, key, values):
        """
        Tính xác suất của từng trường hợp
        """
        scores = defaultdict(float)
        all_inst = 0 
        for v in values: 
            if len(v[0]) == 2: 
                scores[v[0][-1]] = v[1]
                all_inst += v[1]

        scores_prob = {k: v/all_inst for k, v in scores.items()}
        prob = defaultdict(float)
        for v in values: 
            if len(v[0]) == 3:
                prob[v[0]] = v[1] / scores[v[0][-1]]

        # Ensure 'yes' and 'no' keys exist in scores_prob
        for i in ['yes', 'no']: 
            if i in scores_prob:
                scores_prob[i] = math.log(scores_prob[i])
            else:
                scores_prob[i] = float('-inf')  # Assign a very low probability if the key is missing

        for i in ['yes', 'no']: 
            for k, v in prob.items(): 
                if k[-1] == i:
                    scores_prob[i] += math.log(v)
        
        yield key, max(scores_prob, key=scores_prob.get)



    def steps(self): 

        return [
            MRStep( mapper_init = self.mapper_init,
                    mapper = self.mapper_train,
                    reducer = self.reducer_train), 
            MRStep( mapper_init = self.mapper_init,
                    mapper = self.mapper_infer,
                   reducer = self.reducer_infer)
        ]



if __name__ == '__main__':
    MRNaiveBayes.run()