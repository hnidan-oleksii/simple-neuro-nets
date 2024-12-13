from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem
)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPen, QBrush, QColor

from numpy import max
from numpy import emath


class NeuralNetworkVisualizer(QGraphicsView):
    def __init__(self, node_radius=150, node_spacing=1000, connection_width=1):
        super().__init__()

        self.node_radius = node_radius
        self.node_spacing = node_spacing
        self.connection_width = connection_width

    def draw_network(self, layers):
        self.layers = layers
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.colors = self.generate_colors(len(self.layers))

        self.scale_factor = emath.logn(0.99, max(layers))
        self.scale_factor = 0.5

        for layer_index, nodes_in_layer in enumerate(self.layers):
            self.draw_layer(layer_index, nodes_in_layer)

        self.scale(self.scale_factor, self.scale_factor)

    def draw_layer(self, layer_index, nodes_in_layer):
        x_offset = layer_index * (self.node_radius*2 + self.node_spacing) + 100
        total_height = \
            (nodes_in_layer * (self.node_radius * 2 + self.node_spacing)) \
            - self.node_spacing
        y_start = 300 - total_height // 2

        node_positions = []
        for node_index in range(nodes_in_layer):
            y_position = y_start + node_index \
                * (self.node_radius * 2 + self.node_spacing)
            self.draw_node(x_offset, y_position, layer_index)
            node_positions.append((x_offset, y_position))

        if layer_index < len(self.layers) - 1:
            next_layer_nodes = self.layers[layer_index + 1]
            self.draw_connections(node_positions, next_layer_nodes, layer_index)

    def draw_node(self, x, y, layer_index):
        node_color = self.colors[layer_index]
        ellipse = QGraphicsEllipseItem(
            QRectF(x - self.node_radius,
                   y - self.node_radius,
                   self.node_radius * 2,
                   self.node_radius * 2)
        )
        ellipse.setBrush(QBrush(node_color))
        ellipse.setPen(QPen(Qt.black, 1))
        self.scene.addItem(ellipse)

    def draw_connections(self, node_positions, next_layer_nodes, layer_index):
        x_next_layer = (layer_index + 1) \
            * (self.node_radius * 2 + self.node_spacing) \
            + 100
        total_height_next = \
            (next_layer_nodes
             * (self.node_radius * 2 + self.node_spacing)) \
            - self.node_spacing
        y_start_next = 300 - total_height_next // 2

        for i, (x1, y1) in enumerate(node_positions):
            for j in range(next_layer_nodes):
                y2 = y_start_next + j * \
                        (self.node_radius * 2 + self.node_spacing)
                line = QGraphicsLineItem(x1, y1, x_next_layer, y2)
                line.setPen(QPen(Qt.black, self.connection_width))
                self.scene.addItem(line)

    def generate_colors(self, num_layers):
        """Generate distinct colors for each layer."""
        colors = []
        for i in range(num_layers):
            hue = int(360 * i / num_layers)
            colors.append(QColor.fromHsv(hue, 255, 200))
        return colors

    def wheelEvent(self, event):
        """Override the wheel event to implement zoom in and zoom out."""
        if event.angleDelta().y() > 0:
            self.scale(self.scale_factor, self.scale_factor)
        else:
            self.scale(1 / self.scale_factor, 1 / self.scale_factor)
