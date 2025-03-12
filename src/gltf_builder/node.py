'''
Builder representation of a gltr node. This will be compiled down during
the build phase.
'''


from collections.abc import Sequence

from gltf_builder.primitives import Primitive, PrimitiveType, Point


class BNodeContainer:
    children: list['BNode']
    @property
    def nodes(self):
        return self.children
    
    def __init__(self,
                 children: Sequence['BNode']=()
                 ):
        self.children = list(children)
    
    def add_node(self,
                children: Sequence['BNode']=(),
                primitives: Sequence['Primitive']=(),
                ) -> 'BNode':
        node = BNode(children=children, primitives=primitives)
        self.children.append(node)
        return node


class BNode(BNodeContainer):
    name: str
    index: int = -1
    primitives: list[Primitive]
    def __init__(self, name: str ='',
                 primitives: Sequence[Primitive] = (),
                 children: Sequence['BNode']=(),
                 ):
        super().__init__(children)
        self.name = name
        self.primitives = list(primitives)
        
    def add_primitive(self, type: PrimitiveType, *points: Point) -> Primitive:
        prim = Primitive(type, points)
        self.primitives.append(prim)
        return prim
    
