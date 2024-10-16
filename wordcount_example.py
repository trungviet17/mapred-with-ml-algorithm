from mrjob.job import MRJob
from mrjob.step import MRStep
import re

WORD_RE = re.compile(r"[\w']+")


class MRWordCountBasic(MRJob):
    """
    Giải thích code của thư viện : 
        1. Class định nghĩa các hàm được gọi làm step có thể là hàm mapper, reducer hoặc combiner
        2. Các hàm trên được việc đè (overide)
    """
    def mapper(self, _, line):
        """
        Hàm mapper nhận key và value làm input
            1. key ở đây được lược bỏ 
            2. value ở đây được tính là line 
        -> Hàm trả về value và key tương ứng 
        """
        yield "chars", len(line)
        yield "words", len(line.split())
        yield "lines", 1

    def reducer(self, key, values):
        """
        Hàm reduce nhận vào key và values tương ứng và trả về tổng values với mỗi key 
        """
        yield key, sum(values)


class MRMultipleStepWordCount(MRJob): 
    """
    Class dùng để ví dụ về việc thực hiện nhiều step trong một job
    """


    def mapper_get_words(self, _, line): 
        # mapp tất các các chuỗi dữ liệu từ file, trả về mỗi từ từ file  
        for word in WORD_RE.findall(line): 
            yield (word.lower(), 1)


    def combiner_count_words(self, word, counts): 
        # trả về tổng số lần xuất hiện của mỗi từ 
        yield (word, sum(counts))



    def reducer_count_words(self, word, counts): 
        yield None, (sum(counts), word)

    
    def reducer_find_max_word(self,_ , wordcount_pairs):
        yield max(wordcount_pairs)
        


    def steps(self): 

        return [
            MRStep(mapper=self.mapper_get_words,
                    combiner=self.combiner_count_words,
                    reducer=self.reducer_count_words),
            MRStep(reducer=self.reducer_find_max_word)
        ]



class MRWordFreqCount(MRJob):
    """
    Class ví dụ về việc sử dụng hàm init, final trong mapper và reducer
    """
    def init_get_words(self):
        self.words = {}

    def get_words(self, _, line):
        for word in WORD_RE.findall(line):
            word = word.lower()
            self.words.setdefault(word, 0)
            self.words[word] = self.words[word] + 1

    def final_get_words(self):
        for word, val in self.words.iteritems():
            yield word, val

    def sum_words(self, word, counts):
        yield word, sum(counts)

    def steps(self):
        return [MRStep(mapper_init=self.init_get_words,
                       mapper=self.get_words,
                       mapper_final=self.final_get_words,
                       combiner=self.sum_words,
                       reducer=self.sum_words)]




if __name__ == '__main__':
    MRMultipleStepWordCount.run()