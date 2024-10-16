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
    def mapper_train(self, _, line):
        line = line.strip().split(',') 
        target = line[-1] 
        label = ['outlook', 'temp', 'humidity', 'wind']
        for i in range(len(line) -1) : 
            yield (label[i], line[i], target) , 1


    def reducer_train(self, key, value): 
        yield key, sum(value)

    

    def init_mapper_infer(self):
        """
        prob : dùng để lưu xác xuất theo dạng p(featủe = value | target)
        counts : lưu số lần xuất hiện target -> tính p(target)
        """
        self.prob = defaultdict( lambda: defaultdict(lambda: defaultdict(int)))
        self.counts = defaultdict(int) 


    def mapper_infer(self, key, value): 
        label, feature_value, target = key
        self.counts[target] += value
        self.prob[label][feature_value][target] += value
    

    def reducer_infer(self, _, line):
        line = line.strip().split(',')
        label = ['outlook', 'temp', 'humidity', 'wind']
        targets = list(self.counts.keys())

        score = defaultdict(float)
        for target in targets: 
            score[target] = math.log(self.counts[target] / sum(self.counts.values()))

            for i in range(len(line)): 
                score[target] += math.log(self.prob[label[i]][line[i]][target] / self.counts[target])
        result = max(score, key = score.get)
        yield line, result

    


    def steps(self): 

        return [
            MRStep(mapper = self.mapper_train,
                    reducer = self.reducer_train), 
            MRStep(mapper_init = self.init_mapper_infer,
                    mapper = self.mapper_infer,
                    reducer = self.reducer_infer)
        ]



if __name__ == '__main__':
    MRNaiveBayes.run()
    pass 