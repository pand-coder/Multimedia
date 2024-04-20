import cv2
import numpy as np
from copy import deepcopy as cp

class Node:
    def __init__(self, f, p, isLeaf=0):
        self.freq = f          # For storing intensity
        self.prob = p          # For storing Probability
        self.word = ""         # For coded word ex : '100101'
        # Pointers for children, C[0] for left child, c[1] for right child
        self.c = [None, None]
        self.isLeaf = isLeaf   # Flag for Leaf-Nodes

class Image:
    def __init__(self):
        self.path_in = ""               # Input file Location
        self.path_out = ""              # Output file Location
        self.im = np.zeros(1)           # Input Image
        self.out = np.zeros(1)          # Output Image
        self.image_data = np.zeros(1)   # List of Intensities and dimensions
        self.r = 0                      # Rows
        self.c = 0                      # Coloums
        self.d = 0                      # Depth / channels

        # histogram, ie frequency count of each data value
        self.hist = np.zeros(1)
        self.freqs = np.zeros(1)        # For Non-zero frequency

        # Dictionary with key as freqs and values as Probailities
        self.prob_dict = {}
        self.allNodes = []              # Container for  All created Nodes
        self.leafNodes = {}             # Container for Leaf Nodes
        self.root = Node(-1, -1)        # Root Node with Probability = 1

        # Encoded String of Image,form: "01001010101011......", interpretation :  [r,c,d,[..pxls]]
        self.encodedString = ""

        # Decoded List of integers when read from .bin file, form : [456,342,3,34,2,120,44, ...... ], interpretation : [r,c,d,[..pxls]]
        self.decodeList = []

        # Binary from file in for of integers ie [0,1,0,0,1,0,1,0,1,0,1,0,1,1,......]
        self.binaryFromFile = []

    def checkCoding(self):
        return np.all(self.im == self.out)

    def readImage(self, path):
        self.path_in = path
        try:
            self.im = cv2.imread(path)
        except:
            print("Error in reading image")

    def initialise(self):
        self.r, self.c, self.d = self.im.shape

        # Pushing r,c,d to encode into image_data list
        temp_list = self.im.flatten()
        temp_list = np.append(temp_list, self.r)
        temp_list = np.append(temp_list, self.c)
        temp_list = np.append(temp_list, self.d)

        self.image_data = temp_list

        # Creating historgram from image_data to create frequencies.
        self.hist = np.bincount(
            self.image_data, minlength=max(256, self.r, self.c, self.d))
        total = np.sum(self.hist)

        # Extracting the non-zero frequencies
        self.freqs = [i for i, e in enumerate(self.hist) if e != 0]
        self.freqs = np.array(self.freqs)

        # Creatn=ing a dict of propabilities , with keys are intensities and value as propabilities
        for i, e in enumerate(self.freqs):
            self.prob_dict[e] = self.hist[e]/total

    def outImage(self, path):
        self.path_out = path
        try:
            cv2.imwrite(self.path_out, self.out)
        except:
            print("Error in writing the image")

    def buildNodes(self):
        for key in self.prob_dict:
            leaf = Node(key, self.prob_dict[key], 1)
            self.allNodes.append(leaf)

    def prob_key(self, e):
        return e.prob

    def upTree(self):
        import heapq
        self.buildNodes()

        # Sorting all Nodes in workspace to create uptree
        workspace = sorted(cp(self.allNodes), key=self.prob_key)
        while(1):
            c1 = workspace[0]
            c2 = workspace[1]
            workspace.pop(0)
            workspace.pop(0)

            # Creating A new node from  two smallest propability intensities
            new_node = Node(-1, c1.prob+c2.prob)
            new_node.c[0] = c1
            new_node.c[1] = c2

            workspace = list(heapq.merge(
                workspace, [new_node], key=self.prob_key))   # Pushing the created Node into Workspace
            # Break if probability of prepared node is 1, indicating preparing upTree is completed
            if(new_node.prob == 1.0):
                self.root = new_node        # And storing it as root Node
                return

    def downTree(self, root, word):
        root.word = word
        if(root.isLeaf):
            self.leafNodes[root.freq] = root.word
        if(root.c[0] != None):
            self.downTree(root.c[0], word+'0')
        if(root.c[1] != None):
            self.downTree(root.c[1], word+'1')

    def huffmanAlgo(self):
        self.upTree()                   # Creating UpTree
        self.downTree(self.root, "")    # Creating DownTree

        dicti = {}                          # Storing the prob_dict in new variable dicti
        # So that we need not access ("self.") every time that costs time, we just use dicti in place of self.leafNodes
        for key in self.leafNodes:
            dicti[key] = self.leafNodes[key]

        # Storing the self.encodedString in new variable encodedString
        # So that we need not accecess "self." every time,which cost more time
        encodedString = ""
        encodedString += dicti[self.r]
        encodedString += dicti[self.c]
        encodedString += dicti[self.d]

        # Note we are first encoding dimensions, and later encoding each pxl in 3rd dimension order , later while decoding we decode in the same way

        for i in range(self.r):
            for j in range(self.c):
                for ch in range(self.d):
                    encodedString += dicti[self.im[i][j][ch]]

        self.encodedString = encodedString

    def sendBinaryData(self, path):
        from bitstring import BitArray
        file = open(path, 'wb')
        obj = BitArray(bin=self.encodedString)
        obj.tofile(file)
        file.close()

    def decode(self, path):
        import bitarray
        self.binaryFromFile = bitarray.bitarray()
        with open(path, 'rb') as f:
            self.binaryFromFile.fromfile(f)

        decodeList = []
        root = self.root
        temp_root = cp(self.root)

        temp_r = 0
        temp_c = 0
        temp_d = 0

        for i, c_int in enumerate(self.binaryFromFile):
            if(temp_r != 0 and temp_c != 0 and temp_d != 0 and len(decodeList) == (temp_r*temp_c*temp_d + 3)):
                break
            if(temp_r == 0 and len(decodeList) >= 1):
                temp_r = decodeList[0]

            if(temp_c == 0 and len(decodeList) >= 2):
                temp_c = decodeList[1]

            if(temp_d == 0 and len(decodeList) >= 3):
                temp_d = decodeList[2]

            temp_root = temp_root.c[c_int]
            if(temp_root.isLeaf):
                decodeList.append(temp_root.freq)
                temp_root = root
                continue

        self.decodeList = decodeList

    def decodeIm(self, path):
        self.decode(path)

        decodeList = self.decodeList
        out_r = decodeList[0]
        decodeList.pop(0)
        out_c = decodeList[0]
        decodeList.pop(0)
        out_d = decodeList[0]
        decodeList.pop(0)

        out = np.zeros((out_r, out_c, out_d))

        for i in range(len(decodeList)):
            id = i//out_d
            x = id//out_c
            y = id % out_c
            z = i % out_d
            out[x][y][z] = decodeList[i]
        out = out.astype(dtype=int)
        self.out = out

    def huffmanCode(self, input_path, compressed_path, output_path, toCheck=0):
        self.readImage(input_path)
        print("Read the input image successfully.")

        self.initialise()
        print("Initialized image data.")

        print("Encoding image using Huffman coding...")
        self.huffmanAlgo()
        print("Image encoding complete.")

        print("\nOriginal Size of Image : ", self.r*self.c*self.d*8, " bits")
        print("Compressed Size : ", len(self.encodedString), " bits")
        print("Compressed factor : ", self.r * self.c*self.d*8 / len(self.encodedString), "\n")

        print("Sending coded data...")
        self.sendBinaryData(compressed_path)
        print("Coded data sent.")

        print("Decoding compressed image...")
        self.decodeIm(compressed_path)
        self.outImage(output_path)
        print("Decoded compressed image. Open output image from the above path.")

        if toCheck:
            print("Are both images same : ", self.checkCoding())

image = Image()
image.huffmanCode("lake.png", "compressed.bin", "output.png", toCheck=1)
