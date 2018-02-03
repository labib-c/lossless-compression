"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for char in text:
        if char not in freq_dict:
            freq_dict[char] = 1
        else:
            freq_dict[char] += 1
    return freq_dict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    tree_list = []
    if len(freq_dict) > 0:
        for key in freq_dict:   # list of tuples(HNode, it's frequency)
            node = HuffmanNode(key)
            freq = freq_dict[key]
            tree_list.append((node, freq))
    else:
        return HuffmanNode()
    if len(tree_list) > 1:
        while len(tree_list) > 1:
            freq = 0
            low1 = lowest_priority(tree_list)
            freq += low1[1]
            tree_list.remove(low1)
            low2 = lowest_priority(tree_list)
            freq += low2[1]
            tree_list.remove(low2)
            new_node = HuffmanNode(None, low1[0], low2[0])
            tree_list.append((new_node, freq))
    else:
        new_node = HuffmanNode(None, tree_list[0][0], HuffmanNode('x'))
        return new_node  # dummy variable as HTree must have two or 0 children
    return tree_list[0][0]


def lowest_priority(l):
    """ Return node from l list of (trees, frequency) which has the lowest
    frequency value

    @type l: list of ((HuffmanNode, int))
    @rtype: tuple of (HuffmanNode, int)
    """
    lowest = l[0]
    for t in l:
        if t[1] < lowest[1]:
            lowest = t
    return lowest


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    code_dict = {}
    if tree.is_leaf():
        code_dict[tree.symbol] = ""
        return code_dict
    else:
        left_dict = {}
        right_dict = {}
        if tree.left:
            left_dict = get_codes(tree.left)
            for item in left_dict:
                left_dict[item] = "0" + left_dict[item]
        if tree.right:
            right_dict = get_codes(tree.right)
            for item in right_dict:
                right_dict[item] = '1' + right_dict[item]
        code_dict.update(left_dict)
        code_dict.update(right_dict)
        return code_dict


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    count(tree)   # helper method used number_nodes returns NoneType
    # and we had to return num for the function to work


def count(tree, num=0):
    """ Return the assigned number value num of the internal node of Huffman
     Tree tree.

    @type tree: HuffmanNode
    @type num: int
    @rtype: int
    """
    if tree is not None:
        num = count(tree.left, num)
        num = count(tree.right, num)
        if not tree.is_leaf():
            tree.number = num
            num += 1
        return num
    else:
        return num


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    code_dict = get_codes(tree)
    total = 0
    for key in code_dict:
        total += len(code_dict[key])*freq_dict[key]
    total_symbols = 0
    for symbols in freq_dict:
        total_symbols += freq_dict[symbols]
    return total / total_symbols


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    all_bits = ''
    for symbol in text:
        all_bits += codes[symbol]

    while len(all_bits) % 8:
        all_bits += '0'

    all_bytes = []
    i = 8
    for _ in range(len(all_bits) // 8):
        all_bytes.append(bits_to_byte(all_bits[i-8:i]))
        i += 8
    return bytes(all_bytes)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    # first add 0 if leaf, 1 if node on left
    # second add symbol/number
    # third add 0 for leaf, 1 if node on right
    # fourth add symbol/number
    # done in post-order
    if tree is None:
        return bytes([])

    elif tree.is_leaf():
        return bytes([])

    elif tree.left.is_leaf() and tree.right.is_leaf():
        return bytes([0, tree.left.symbol,
                      0, tree.right.symbol])

    elif tree.left.is_leaf() and not tree.right.is_leaf():
        return (tree_to_bytes(tree.right) +
                bytes([0, tree.left.symbol] +
                      [1, tree.right.number]))

    elif tree.right.is_leaf() and not tree.left.is_leaf():
        return (tree_to_bytes(tree.left) +
                bytes([1, tree.left.number] +
                      [0, tree.right.symbol]))

    else:
        return (tree_to_bytes(tree.left) +
                tree_to_bytes(tree.right) +
                bytes([1, tree.left.number] +
                      [1, tree.right.number]))


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    root_data = node_lst[root_index]
    root = HuffmanNode()

    # root node is list [root_index]
    # if root_nodes l_type is 0, then the roots left is root node.l_data
    # if root_nodes l_type is 1, then go into function recursively
    # with the new index being the root_nodes.l_type
    # mirror this on right side
    if not root_data.l_type:
        root.left = HuffmanNode(root_data.l_data)

    if root_data.l_type:
        root.left = generate_tree_general(node_lst, root_data.l_data)

    if not root_data.r_type:
        root.right = HuffmanNode(root_data.r_data)

    if root_data.r_type:
        root.right = generate_tree_general(node_lst, root_data.r_data)

    return root


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    new_lst = [node_lst[i] for i in range(root_index + 1)]
    stack = Stack()

    while new_lst:
        node_data = new_lst.pop(0)

        if not node_data.l_type and not node_data.r_type:
            left = HuffmanNode(node_data.l_data)
            right = HuffmanNode(node_data.r_data)
            stack.add(HuffmanNode(None, left, right))

        elif not node_data.l_type:
            left = HuffmanNode(node_data.l_data)
            right = stack.pop()
            stack.add(HuffmanNode(None, left, right))

        elif not node_data.r_type:
            right = HuffmanNode(node_data.r_data)
            left = stack.pop()
            stack.add(HuffmanNode(None, left, right))

        elif node_data.r_type and node_data.l_type:
            right = stack.pop()
            left = stack.pop()
            stack.add(HuffmanNode(None, left, right))

    while not stack.is_empty():
        node1 = stack.pop()
        if stack.is_empty():
            return node1
        node2 = stack.pop()
        stack.add(HuffmanNode(None, node2, node1))


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    all_bytes = []
    all_bits = ""
    curr_node = tree
    i = 0

    for byte in text:
        all_bits += byte_to_bits(byte)

    while len(all_bytes) != size:
        if not curr_node.left and not curr_node.right:
            all_bytes.append(curr_node.symbol)
            curr_node = tree

        elif all_bits[i] is '1':
            curr_node = curr_node.right
            i += 1

        elif all_bits[i] is '0':
            curr_node = curr_node.left
            i += 1
    return bytes(all_bytes)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))

# ====================
# Other functions


def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    priority_queue = PriorityQueue()
    for symbol in freq_dict:
        priority_queue.enqueue((HuffmanNode(symbol), freq_dict[symbol]))
    stack = Stack()
    while not priority_queue.is_empty():
        stack.add(priority_queue.dequeue()[0])

    queue = Queue()
    queue.enqueue(tree)
    while not queue.is_empty():
        node = queue.dequeue()
        if not node.left and not node.right:
            efficient_node = stack.pop()
            node.symbol = efficient_node.symbol
        else:
            queue.enqueue(node.left)
            queue.enqueue(node.right)


class Stack:
    """
    Last-in, first-out (LIFO) stack.
    """

    def __init__(self):
        """
        Create a new, empty Stack self.

        @param Stack self: this stack
        @rtype: None
        """
        self._stack = []

    def add(self, obj):
        """
        Add object obj to top of Stack self.

        @param Stack self: this Stack
        @param object obj: object to place on Stack
        @rtype: None
        """
        self._stack.append(obj)

    def pop(self):
        """
        Remove and return top element of Stack self.

        Assume Stack self is not empty.

        @param Stack self: this Stack
        @rtype: object

        >>> s = Stack()
        >>> s.add(5)
        >>> s.add(7)
        >>> s.pop()
        7
        """
        return self._stack.pop()

    def is_empty(self):
        """
        Return whether Stack self is empty.

        @param Stack self: this Stack
        @rtype: bool
        """
        return len(self._stack) == 0


class Queue:
    """
    First-in, first out stack.
    """
    def __init__(self):
        """(Queue) -> None

        Initialize an empty Queue.
        """
        self._queue = []

    def enqueue(self, obj):
        """(Queue, object) -> None
        Add an object to this Queue
        """
        self._queue.append(obj)

    def dequeue(self):
        """(Queue) -> object
        Remove and return the bottom most object in this Queue
        """
        return self._queue.pop(0)

    def is_empty(self):
        """(Queue) -> bool
        Return True iff no more objects are left in this Queue
        """
        return len(self._queue) == 0


class PriorityQueue:
    """Remove objects according to their priority.
    """
    def __init__(self):
        """(PriorityQueue) -> None
        Initialize a queue with zero items
        """
        self._queue = []

    def enqueue(self, obj):
        """"(PriorityQueue, tuple(HuffmanNode, int)) -> None
        Add obj to this PriorityQueue
        """
        self._queue.append(obj)

    def dequeue(self):
        """(PriorityQueue) -> tuple
        Remove and return the object with most priority
        """
        all_freq = []
        for item in self._queue:
            all_freq.append(item[1])

        for item in self._queue:
            if item[1] == min(all_freq):
                self._queue.remove(item)
                return item

    def is_empty(self):
        """(PriorityQueue) -> bool
        Return True iff no more objects are left in
        this Priority Queue.
        """
        return len(self._queue) == 0


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
