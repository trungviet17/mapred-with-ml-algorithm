from mrjob.job import MRJob
from  mrjob.step import MRStep
import random 



class MRKmeans(MRJob): 

    def configure_args(self): 
        """
        thêm đường dẫn tới file centroids vào cmd 
        """
        super(MRKmeans, self).configure_args()
        self.add_file_arg('--centroids', help = 'path to kmeans.txt')


    def load_centroids(self): 
        
        centroids = []
        with open(self.options.centroids, 'r') as file: 
            for line in file: 
                line = line.strip()
                _, x, y = line.split(',')
                centroids.append((float(x), float(y)))
        return centroids
    
    def mapper_init(self): 
        """
        Khởi tạo centroids từ file 
        """
        self.centroids = self.load_centroids()

    
    def mapper(self, _, line): 
        _, x, y = line.strip().split(',')
        x = float(x)
        y = float(y)

        # tính khoảng cách euclid giữa các centroids
        new_centroid = min(self.centroids, key = lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
        yield new_centroid, (x, y)

    def reducer(self, centroid, points):
        x = 0
        y = 0
        count = 0
        for p in points: 
            x += p[0]
            y += p[1]
            count += 1
            
        # trả về key là centroids cũ và value là centroids mới
        yield centroid, (x/count, y/count)

    def step(self): 
        return [
            MRStep(mapper_init = self.mapper_init, 
                    mapper = self.mapper, 
                    reducer = self.reducer)
        ]



if __name__ == "__main__":


    def generate_data(num_points = 10, file_path = 'data.txt'): 
        data = []
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        
        with open(file_path, 'w') as file:
            for i in range(num_points):
                label = labels[i % len(labels)]
                x = random.randint(0, 20)
                y = random.randint(0, 20)
                file.write(f"{label}, {x}, {y}\n")
                data.append((label, x, y))
        
        return data
    
    MRKmeans.run()

