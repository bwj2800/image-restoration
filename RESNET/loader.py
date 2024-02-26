import torchvision

# target_transform 대신 __getitem__ 메서드를 사용하여 이미지의 경로를 추출
class CustomImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(path)
        return sample, target


