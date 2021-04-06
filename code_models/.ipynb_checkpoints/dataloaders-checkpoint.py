
class RBE_dataset(Dataset):
    def __init__(self, data_dir, transform=None,head_root=''):
        self.data_dir=data_dir
        self.imgs_list=os.listdir(data_dir)
        self.transform=transform
        self.head_root=head_root

    def __getitem__(self, index):

        image_path = os.path.join(self.data_dir,self.imgs_list[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image_t = self.transform(image)

        return image_t, image_path.replace(self.head_root,'')

    def __len__(self):
        return len(self.imgs_list)

#