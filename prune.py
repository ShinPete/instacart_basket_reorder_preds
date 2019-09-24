'''
    Module: prune.py
    Author: Matthew McGonagle
    
    Classes and functions to do cost-complexity pruning on a sci-kit learn DecisionTreeClassifier. As described in Hastie, Tibshirani, and Friedman's
    "The Elements of Statistical Learning", one method to prevent overfitting of a decision tree classifier is to grow the tree very far, and then
    prune it back in such a way that we add in a weighted cost for the size of the tree. This method allows different branches of the tree to
    grow to different levels while preventing the tree from being overly complex. Hence, the size of the tree is meant to be a measure of the tree's
    complexity, hence the name cost-complexity pruning.

    One can use cross-validation to pick out an appropriate weight.
'''

import numpy as np
from graphviz import Digraph
from matplotlib import cm
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Pruner:
    '''
    Class for doing pruning of a sci-kit learn DecisionTreeClassifier. At initialization, the order of the nodes to prune is found, but no 
    pruning is done. The order of pruning is determined by the pruning that results in the smallest increase in the cost (e.g. entropy or gini index)
    of the tree is done first. Note, a node can only be pruned if both of its children are leaf nodes (also recall that for a sci-kit learn 
    DecisionTreeClassifier the children always come in pairs).

    Members
    -------
    tree : sklearn.tree.Tree
        A reference to the tree object in a sci-kit learn DecisionTreeClassifier; in such a classifier, this member object is usually called tree_.

    leaves : List of Int 
        A list of the indices which are leaf nodes for the original decision tree.

    parents : Numpy array of Int 
        Provides the index of the parent node for nodei. Note, the fact that the root has no parent is indicated by setting
        parents[0] = -1.        
    
    pruneCosts : Numpy array of Float
        The cost increase if nodei is pruned. Note that cost is calculated by weighting a node's impurity (e.g. entropy or gini index) by
        the number of samples in the node.

    originalCost : Float
        The original cost for the fully grown tree, i.e. the total cost for all of the original leaf nodes.

    originalChildren : Pair of Numpy array of Int
        A copy of the original left and right children indices for the original tree. So it is (children_left.copy(), children_right.copy()).

    pruned : Numpy array of Bool
        Used for calculating the prune sequence. Holds whether nodei has been pruned. The leaf nodes are considered to automatically have
        been pruned.

    pruneSequence : Numpy array of Int
        The order to prune the nodes. pruneSequence[0] = -1 to indicate the sequence starts with no pruning; so pruneSequence[i] is the ith node
        to prune. 

    pruneState : Int
        Holds the current number of nodes pruned. So a state of 0 means no pruning has occurred. This is changed with the member function prune().
        Initialized to 0. 
    '''

    def __init__(self, tree):

        '''
        Finds the prune sequence and initializes the prune state to be 0.

        Parameters
        ---------
        tree : sklearn.tree.Tree
            A reference to the tree that will be pruned by this pruner. Note that for sklearn.tree.DecisionTreeClassifier, the tree
            is the member variable DecisionTreeClassifier.tree_.
        '''

        self.tree = tree

        self.leaves = self._getLeaves(tree)
        self.parents = self._getParents(tree)
        self.pruneCosts = self._getPruneCosts(tree)
        self.originalCost = self.pruneCosts[self.leaves].sum()
        self.originalChildren = list(zip(tree.children_left.copy(), tree.children_right.copy()))

        # Initially, only the leaves count as being already pruned.

        self.pruned = np.full(len(tree.impurity), False)
        self.pruned[self.leaves] = True

        self.pruneSequence, self.costSequence = self._makePruneSequence(tree)
        self.pruneState = 0

    def _getLeaves(self, tree):

        '''
        Find the leaf nodes of the tree.
        Parameters
        ----------
        tree : sklearn.tree.Tree
            The tree to find the leaf nodes of.

        Returns
        -------
        List of Int
            The list of indices that correspond to leaf nodes in the tree.
        '''

        leaves = []

        # Note that children always come in pairs.

        for nodei, lChild in enumerate(tree.children_left):

           if lChild == -1:
                leaves.append(nodei) 
       
        return leaves 

    def _getParents(self, tree):
        '''
        Find the list of indices of parents for each node. The parent of the root node is defined to be -1.
        
        Parameters
        ----------
        tree : sklearn.tree.Tree
            Tree to find the list of parents for.

        Returns
        -------
        Numpy array of Int
            The indices of the parent node for each node in the tree. We consider the parent of the root to be -1.
        '''

        parents = np.full(len(tree.children_left), -1) 

        for nodei, children in enumerate(zip(tree.children_left, tree.children_right)):

            lChild, rChild = children

            # Children always come in pairs for a decision tree.

            if lChild != -1:
               parents[lChild] = nodei
               parents[rChild] = nodei

        return parents

    def _getCost(self, tree, nodei):

        '''
        Get the cost of a node; i.e. the product of the impurity and the number of samples. Note, this is not
        the cost of pruning the node.

        Parameters
        ----------
        tree : sklearn.tree.Tree
            The tree that the node is in.
        nodei : Int
            The index of the node to calculate the cost of.

        Returns
        -------
        Float
            The cost of the node (NOT the cost of pruning the node).
        '''

        cost = tree.n_node_samples[nodei] * tree.impurity[nodei]

        return cost

    def _getPruneCosts(self, tree):
        '''
        Calculate the cost of pruning each node. This is the amount that the total cost of the current pruned
        tree will increase by if we prune the node. Is given by the difference between the cost of this node and
        the costs of its children.

        Note, there isn't really any cost associated with pruning a leaf node as they aren't prunable; so they are given a
        cost of 0.

        Parameters
        ----------
        tree : sklearn.tree.Tree
            The original unpruned tree to calculate the costs for.

        Returns
        -------
        Numpy array of Float
            The costs of pruning each node in the tree.
        '''

        pruneCosts = np.zeros(len(tree.impurity))
        nodeCosts = tree.n_node_samples * tree.impurity

        for nodei, (lChild, rChild) in enumerate( zip( tree.children_left, tree.children_right) ):

            # Children always come in pairs.

            if lChild != -1:

                decrease = nodeCosts[nodei] - nodeCosts[lChild] - nodeCosts[rChild] 
                pruneCosts[nodei] = decrease

        return pruneCosts

    def _getInitialCandidates(self, tree):
        '''
        Find the initial list of prunable nodes (i.e. parents whose both left and right children are leaf nodes).
        Also find their prune costs.

        Parameters
        ----------
        tree : sklearn.tree.Tree
            The original unpruned tree.

        Returns
        -------
        List of Int
           The indices of the initial candidates to prune. 
        List of Float
           Their corresponding list of prune costs.
        '''

        candidates = []
        candidateCosts = []

        for leafi in self.leaves:

            parenti = self.parents[leafi]
            if parenti != -1:
                lChild = tree.children_left[parenti]
                rChild = tree.children_right[parenti]

                if self.pruned[lChild] and self.pruned[rChild] and parenti not in candidates:
                    candidates.append(parenti)
                    candidateCosts.append(self.pruneCosts[parenti])

        return candidates, candidateCosts

    def _popNextPrune(self, candidates, costs):
        '''
        Remove the next prune node from the list of candidates, and also remove its cost from the list of costs. 

        The next node to prune is found by minimizing over the costs of all of the candidates.

        Parameters
        ----------
        candidates : List of Int
            The list of indices of nodes that we could potentially prune next.
        costs : List of Float
            The corresponding list of pruning costs for each candidate.

        Returns
        -------
        Int
            The index of the next prune node.
        '''

        minCosti = np.argmin(costs)
        nextPrune = candidates.pop(minCosti)
        costs.pop(minCosti)

        return nextPrune
        

    def _makePruneSequence(self, tree):
        '''
        Find the order to prune the nodes for cost-complexity pruning. The order is determined by the fact that nodes with the smallest
        pruning cost are pruned first. Also find the accumulative pruning cost for pruning in this order.

        Note that pruneSequence[0] = -1, indicating the no pruning. Also costSequence[0] = 0 as no pruning has occured. 
        Parameters
        ----------
        tree : sklearn.tree.Tree
            The original unpruned tree.

        Returns
        -------
        Numpy array of Int
            The order to prune the nodes.

        Numpy array of Float
            The total accumulative pruning cost for pruning the nodes in order.
        '''

        pruneSequence = [-1]
        costSequence = [0]
        currentCost = 0

        candidates, costs = self._getInitialCandidates(tree)

        while candidates:

            prunei = self._popNextPrune(candidates, costs)
            self.pruned[prunei] = True
            pruneSequence.append(prunei)
            currentCost += self.pruneCosts[prunei]
            costSequence.append(currentCost)

            parenti = self.parents[prunei]
            if parenti != -1:
                lChild = tree.children_left[parenti]
                rChild = tree.children_right[parenti]

                if self.pruned[lChild] and self.pruned[rChild]:
                    candidates.append(parenti)
                    costs.append(self.pruneCosts[parenti])

        return np.array(pruneSequence), np.array(costSequence)

    def prune(self, prunei):
        '''
        Do pruning/unpruning on the tree. Technically, pruning is done on splits (and not on nodes).
        We specify the number of split to prune away from the ORIGINAL tree.

        If the number of splits to prune is greater than what we have pruned so far, we prune off
        more splits. If it is less, then we unprune (i.e. restore) splits.
        Parameters
        ----------
        prunei : Int
            The number of splits to prune off the original tree. Negative values specify offset
            from the maximum number of prunes possible, similar to how negative indexing of
            arrays works. 

        '''

        nPrunes = len(self.pruneSequence)

        if prunei < 0:
            prunei += nPrunes  

        # If the new state involves more prunes than the old state, we have to prune nodes.
        # Else we need to restore children to their old state.

        if prunei > self.pruneState:

            for prune in range(self.pruneState + 1, prunei + 1):
                nodei = self.pruneSequence[prune]
                self.tree.children_left[nodei] = -1
                self.tree.children_right[nodei] = -1

        elif prunei < self.pruneState: 

            for prune in range(prunei + 1, self.pruneState + 1):
                nodei = self.pruneSequence[prune]
                lChild, rChild = self.originalChildren[nodei]
                self.tree.children_left[nodei] = lChild
                self.tree.children_right[nodei] = rChild

        # Update the prune state.

        self.pruneState = prunei

    def costComplexity(self, complexityWeight):
        '''
        Compute the cost-complexity curve for a given weight of the complexity. The complexity is simply the number
        of nodes in the pruned tree. So the cost-complexity is a combination of the cost of the tree and the weighted
        size. To the find the optimal complexity weight, one can do something such as cross-validation.
        
        Also, return a list of the sizes for each cost-complexity.

        Paramters
        ---------
        complexityWeight : Float         
            The weight to apply to the complexity measure.

        Returns
        -------
        Numpy of Int
            The size of the pruned tree for each cost-complexity.

        Numpy of Float
            The cost-complexity measure for each tree size.
        '''

        nPrunes = len(self.pruneSequence)

        # Recall that each prune removes two nodes.
        sizes = self.tree.node_count - 2 * np.arange(0, nPrunes)

        costs = np.full(len(sizes), self.originalCost)
        costs += self.costSequence 
        costComplexity = costs + complexityWeight * sizes

        return sizes, costComplexity

    def pruneForCostComplexity(self, complexityWeight):
        '''
        Prune the tree to the minimal cost-complexity for the given provided weight.

        Parameters
        ----------
        complexityWeight : Float
            The complexity weight to use for calculating cost-complexity.
        '''

        sizes, costComplexity = self.costComplexity(complexityWeight)

        minI = np.argmin(costComplexity)

        self.prune(minI)

class Box:
    '''
    Class to keep track of the xy-rectangle that a node in a decision tree classifier applies to.

    Can be used to accurately and precisely draw the output of a decision tree classifier.

    Members
    -------
    lowerBounds : List of Float of size 2.
        Holds the lower bounds of the x and y coordinates.

    upperBounds : List of Float of size 2.
        Holds the upper bounds of the x and y coordinates.

    value : None or Int
        If desired, one can specify the value that the tree is supposed to resolve the node to. 
    '''

    def __init__(self, lowerBounds, upperBounds, value = None): 
        '''
        Initialize the member variables.
        Parameters
        ----------
        lowerBounds : List of Float of size 2
            Holds the lower bounds of the x and y coordinates.

        upperBounds : List of Float of size 2
            Holds the upper bounds of the x and y coordinates.

        value : The value of a node that the box represents. Default is None. 
        '''

        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.value = value

    def split(self, feature, value):
        '''
        Split the box into two boxes specified by whether x or y coordinate (i.e. feature) and the threshold value to split at.
        This corresponds to how a node in a classifying tree splits on a feature and threshold value.

        Parameters
        ----------
        feature : Int
            0 for x-coordinate or 1 for y-coordinate.

        value : Float
            The threshold value to do the split. The left box is feature less than or equal to value. The right box is the feature
            greater than this value.

        Returns
        -------
        Box
            This is the left box corresponding to the left child of a split node in a decision tree classifier.

        Box
            This is the right box corresponding to the right child of a split node in a decision tree classifier. 
        '''

        newUpper = self.upperBounds.copy()
        newUpper[feature] = value

        newLower = self.lowerBounds.copy()
        newLower[feature] = value

        return Box(self.lowerBounds, newUpper), Box(newLower, self.upperBounds)

    def _color(self):
        '''
        Get the color for this box using a colormap.

        Returns
        -------
        Color 
            The colormap output. 
        '''

        cmap = cm.get_cmap("winter")

        if self.value == None:
            return cmap(0.0)

        return cmap(float(self.value))

    def convertMatPlotLib(self, edges = False):
        '''
        Convert the box a matplotlib.patches.Rectangle.

        Parameters
        ----------
        edges : Bool
            Whether to include edges in the rectangle drawing. Default is False.

        Returns
        -------
        matplotlib.patches.Rectangle
            The matplotlib patch for this box.
        '''

        width = self.upperBounds[0] - self.lowerBounds[0]
        height = self.upperBounds[1] - self.lowerBounds[1]

        kwargs = {'xy' : (self.lowerBounds[0], self.lowerBounds[1]),
                  'width' : width,
                  'height' : height,
                 } 
        if edges:
            kwargs['facecolor'] = self._color()
        else:
            kwargs['color'] = self._color()

        return Rectangle(**kwargs)

        
def getLeafBoxes(tree, lowerBounds, upperBounds):
    ''' 
    Get a list of Box for the leaf nodes of a tree and the initial bounds on the output.

    Paramters
    ---------
    tree : sklearn.tree.Tree
        The tree that determines how to split the boxes to find the boxes corresponding to the
        leaf nodes.
    lowerBounds : List of Float of Size 2
        The initial lower bounds on the xy-coordinates of the box.

    upperBounds : List of Float of Size 2
        The inital upper bounds on the xy-coorinates of the box.

    Returns
    -------
    List of Box
        A list of Box for each leaf node in the tree.
    '''

    rootBox = Box(lowerBounds, upperBounds)

    boxes = [(0, rootBox)]
    leaves = []

    # Keep splitting boxes until we are at the leaf level. Use the thresholds and features contained in
    # the tree to do the splitting.

    while boxes:

        nodei, box = boxes.pop()
        lChild = tree.children_left[nodei]

        # If there are no children then we are at a leaf; recall that children always come in pairs for a decision
        # tree.

        if lChild == -1:

            box.value = np.argmax(tree.value[nodei])
            leaves.append(box)

        else:

            rChild = tree.children_right[nodei]

            lBox, rBox = box.split(tree.feature[nodei], tree.threshold[nodei]) 
            boxes.append((lChild, lBox))
            boxes.append((rChild, rBox))

    return leaves

def plotTreeOutput(axis, tree, lowerBounds, upperBounds, edges = False):
    '''
    Get a precise and accurate plot of the output of a decision tree inside a given box.

    Parameters
    ----------
    axis : pyplot axis
        The axis to plot to.

    tree : sklearn.tree.Tree
        The tree to plot.

    lowerBounds : List of Float of Size 2
        The lower bounds of the xy-coordinates of the box to graph.

    upperBounds : List of Float of Size 2
        The upper bounds of the xy-coordinates of the box to graph.

    edges : Bool
        Whether to include edges of the leaf boxes in the output. Default is False.
    '''

    # The output is determined by the leaf nodes of the decision tree.

    leafBoxes = getLeafBoxes(tree, lowerBounds, upperBounds) 
    
    for box in leafBoxes:
    
        rect = box.convertMatPlotLib(edges)
        axis.add_patch(rect)

def makeSimpleGraphViz(tree):
    ''' 
    Make a simple graphviz vizualization for a decision tree. For each node, we simply
    output the index of the node in the tree, and if the node isn't a leaf, then its
    pruning cost.

    Parameters
    ----------
    tree : sklearn.tree.Tree
        The tree to visualize.

    Returns
    -------
    Digraph
        Reference to the graphviz object created.
    '''

    g = Digraph('g')

    # Make nodes

    nodes = [0]

    while nodes:

        node = nodes.pop()

        lChild = tree.children_left[node]
        rChild = tree.children_right[node]

        # Non-leaf nodes contain information on the cost of pruning away their split.

        if lChild != -1:
            costDecrease = tree.impurity[node] * tree.n_node_samples[node]
            costDecrease -= tree.impurity[lChild] * tree.n_node_samples[lChild]
            costDecrease -= tree.impurity[rChild] * tree.n_node_samples[rChild] 
            costDecrease = "\n" + ("% .1f" % costDecrease) 

            nodes.append(lChild)
            nodes.append(rChild) 
        else:
            costDecrease = ""
        g.node(str(node), str(node) + costDecrease) 

    # Make edges

    for node, children in enumerate(zip(tree.children_left, tree.children_right)):

        lchild, rchild = children 

        if lchild != -1:
            g.edge(str(node), str(lchild))

        if rchild != -1:
            g.edge(str(node), str(rchild))

    return g

def makePrunerGraphViz(pruner):
    '''
    Make a simple graphviz vizualization of a pruner's state of pruning a decision tree. For each node, 
    we simply output its index and its prune cost (if it isn't a leaf node of the original tree). Also,
    we highlight active nodes in green, and unactive nodes (i.e. pruned away) in red.

    Parameters
    ----------
    pruner : Pruner
        The pruner attached to the pruned tree that we wish to visualize.

    Returns
    -------
    Digraph
        Reference to the graphviz object for the visualization.
    '''

    tree = pruner.tree
    g = Digraph('g') #, filename = filename)

    # Make nodes

    for node, ((lChild, rChild), newChild, parent) in enumerate(zip(pruner.originalChildren, tree.children_left, pruner.parents)):

        # The root node never has a pruned parent.

        if parent != -1:
            parentPruned = tree.children_left[parent] == -1
        else:
            parentPruned = False

        nodePruned = newChild == -1

        # Non-leaf nodes (in the original tree) contain information on the cost of pruning away their split.

        if lChild != -1:
            costDecrease = tree.impurity[node] * tree.n_node_samples[node]
            costDecrease -= tree.impurity[lChild] * tree.n_node_samples[lChild]
            costDecrease -= tree.impurity[rChild] * tree.n_node_samples[rChild] 
            costDecrease = "\n" + ("% .1f" % costDecrease) 
        else:
            costDecrease = ""

        # Active nodes are green and non-active nodes are red. Non-active includes nodes that have
        # been pruned, but are still leaves in the prune tree.

        if parentPruned:
            g.node(str(node), str(node) + costDecrease, color = "red", style = 'filled')
   
        else:
            g.node(str(node), str(node) + costDecrease, color = "green", style = 'filled')

    # Make edges

    for node, children in enumerate(pruner.originalChildren):

        lchild, rchild = children 

        if lchild != -1:
            g.edge(str(node), str(lchild))

        if rchild != -1:
            g.edge(str(node), str(rchild))

    return g

def doCrossValidation(model, x, y, nCrossVal, weights):
    '''
    Do cross validation for different complexity weights. Use the results to determine
    the best weight to use. For each weight, this finds the optimal pruning that
    minimizes the cost-complexity for the given complexity weight.

    Paramters
    ---------
    model : sklearn.tree.DecisionTreeClassifier
        The tree model to use.

    x : Numpy Array of Shape (nPoints, 2)
        The dependent variables, i.e. features.

    y : Numpy Array of Shape (nPoints, 1)
        The target class for each data point.

    nCrossVal : Int
        The number of cross validations to do for each weight.

    weights : Numpy array of Float
        The different values of the complexity weights to do cross validation over.

    Returns
    -------
    Numpy Array of Int of Shape (nCrossVal, len(weights))
        The sizes of the optimal trees for each run.
    Numpy Array of Float of Shape (nCrossVal, len(weights))
        The accuracy score of the optimal tree for each run. 
    '''

    scores = []
    sizes = []

    # For each repetition of cross-validation, we iterate over all weights.

    for i in range(nCrossVal):
    
        xtrain, xtest, ytrain, ytest = train_test_split(x, y)
        model.fit(xtrain, ytrain)
        pruner = Pruner(model.tree_)

        # Find the optimal pruning for each weight.
    
        runScores = []
        runSizes = []

        for weight in weights:
            
            treeSizes, costComplexity = pruner.costComplexity(weight)
            minI = np.argmin(costComplexity)
            pruner.prune(minI)
            ypredict = model.predict(xtest)
            acc = accuracy_score(ytest, ypredict)
    
            runScores.append(acc)
            runSizes.append(treeSizes[minI])
    
        scores.append(runScores)
        sizes.append(runSizes)
    
    scores = np.array(scores) 
    sizes = np.array(sizes)
   
    return sizes, scores 
