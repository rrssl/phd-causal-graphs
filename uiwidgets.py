"""
UI widgets.

"""
from panda3d.core import TextNode
from panda3d.core import VBase3
import direct.gui.DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectButton
from direct.gui.DirectGui import DirectFrame
from direct.gui.DirectGui import DirectOptionMenu


class DropdownMenu(DirectOptionMenu):
    def __init__(self, parent=None, **kw):
        # We essentially reuse the __init__ from DirectOptionMenu but remove
        # the parts that are no use for a drop-down menu.
        # NB: all things removed here only impacted setItems.

        # Inherits from DirectButton
        optiondefs = (
            # List of items to display on the popup menu
            ('items',       [],             self.setItems),
            # Background color to use to highlight popup menu items
            ('highlightColor', (.5, .5, .5, 1), None),
            # Extra scale to use on highlight popup menu items
            ('highlightScale', (1, 1), None),
            # Alignment to use for text on popup menu button
            # Changing this breaks button layout
            ('text_align',  TextNode.ALeft, None),
            # Remove press effect because it looks a bit funny
            ('pressEffect',     0,          DGG.INITOPT),
           )
        # Merge keyword options with default options
        self.defineoptions(kw, optiondefs)
        # Initialize superclasses
        DirectButton.__init__(self, parent)
        # Record any user specified frame size
        self.initFrameSize = self['frameSize']
        # This is created when you set the menu's items
        self.popupMenu = None
        self.selectedIndex = None
        self.highlightedIndex = None
        # A big screen encompassing frame to catch the cancel clicks
        self.cancelFrame = self.createcomponent(
            'cancelframe', (), None,
            DirectFrame, (self,),
            frameSize=(-1, 1, -1, 1),
            relief=None,
            state='normal')
        # Make sure this is on top of all the other widgets
        self.cancelFrame.setBin('gui-popup', 0)
        self.cancelFrame.bind(DGG.B1PRESS, self.hidePopupMenu)
        # Default action on press is to show popup menu
        self.bind(DGG.B1PRESS, self.showPopupMenu)
        # Check if item is highlighted on release and select it if it is
        self.bind(DGG.B1RELEASE, self.selectHighlightedIndex)
        # Call option initialization functions
        # NB: calling this function in __init__ with the new subclass as an
        # argument is ESSENTIAL, yet poorly documented.
        # https://www.panda3d.org/forums/viewtopic.php?p=12111#p12111
        self.initialiseoptions(DropdownMenu)
        # Need to call this since we explicitly set frame size
        self.resetFrameSize()

    def setItems(self):
        """Create new popup menu to reflect specified set of items

        Can be used via self['items'] = itemList
        """
        # Remove old component if it exits
        if self.popupMenu is not None:
            self.destroycomponent('popupMenu')
        # Create new component
        self.popupMenu = self.createcomponent('popupMenu', (), None,
                                              DirectFrame,
                                              (self,),
                                              relief=self['relief'],
                                              )
        # Make sure it is on top of all the other gui widgets
        self.popupMenu.setBin('gui-popup', 0)
        if not self['items']:
            return
        # Create a new component for each item
        # Find the maximum extents of all items
        itemIndex = 0
        self.minX = self.maxX = self.minZ = self.maxZ = None
        for item in self['items']:
            c = self.createcomponent(
                'item{}'.format(itemIndex), (), 'item',
                DirectButton, (self.popupMenu,),
                text=item, text_align=TextNode.ALeft,
                text_font=self['text_font'],
                text_scale=self['text_scale'],
                text_fg=(1, 1, 1, 1),  # This option is not saved, but why?
                command=lambda i=itemIndex: self.set(i),
                relief=self['relief'],
                frameColor=self['frameColor'])
            bounds = c.getBounds()
            if self.minX is None:
                self.minX = bounds[0]
            elif bounds[0] < self.minX:
                self.minX = bounds[0]
            if self.maxX is None:
                self.maxX = bounds[1]
            elif bounds[1] > self.maxX:
                self.maxX = bounds[1]
            if self.minZ is None:
                self.minZ = bounds[2]
            elif bounds[2] < self.minZ:
                self.minZ = bounds[2]
            if self.maxZ is None:
                self.maxZ = bounds[3]
            elif bounds[3] > self.maxZ:
                self.maxZ = bounds[3]
            itemIndex += 1
        # Calc max width and height
        self.maxWidth = self.maxX - self.minX
        self.maxHeight = self.maxZ - self.minZ
        # Adjust frame size for each item and bind actions to mouse events
        for i in range(itemIndex):
            item = self.component('item%d' % i)
            # So entire extent of item's slot on popup is reactive to mouse
            item['frameSize'] = (self.minX, self.maxX, self.minZ, self.maxZ)
            # Move it to its correct position on the popup
            item.setPos(-self.minX, 0, -self.maxZ - i * self.maxHeight)
            item.bind(DGG.B1RELEASE, self.hidePopupMenu)
            # Highlight background when mouse is in item
            item.bind(DGG.WITHIN,
                      lambda x, i=i, item=item: self._highlightItem(item, i))
            # Restore specified color upon exiting
            fc = item['frameColor']
            item.bind(DGG.WITHOUT,
                      lambda x, item=item, fc=fc: self._unhighlightItem(
                          item, fc))
        # Set popup menu frame size to encompass all items
        f = self.component('popupMenu')
        f['frameSize'] = (0, self.maxWidth, -self.maxHeight * itemIndex, 0)

        # Set initial item to 0 but don't fire callback
        self.set(0, fCommand=0)

        # Adjust popup menu button to fit all items (or use user specified
        # frame size
        if self.initFrameSize:
            # Use specified frame size
            self['frameSize'] = tuple(self.initFrameSize)
        else:
            # Or base it upon largest item
            self['frameSize'] = (self.minX, self.maxX, self.minZ, self.maxZ)
        # Set initial state
        self.hidePopupMenu()

    def showPopupMenu(self, event=None):
        """Make popup visible.

        Adjust popup position if default position puts it outside of
        visible screen region.
        """
        # Show the menu
        self.popupMenu.show()
        # Make sure its at the right scale
        self.popupMenu.setScale(self, VBase3(1))
        # Compute bounds
        b = self.getBounds()
        fb = self.popupMenu.getBounds()
        # Center menu with button
        xPos = ((b[1] - b[0]) - (fb[1] - fb[0])) / 2.
        self.popupMenu.setX(self, xPos)
        # Set height slightly above the button
        self.popupMenu.setZ(self, (self.maxZ - fb[2])*1.1)
        #  self.popupMenu.setZ(
        #      self, self.minZ + (self.selectedIndex + 1)*self.maxHeight)
        # Make sure the whole popup menu is visible
        pos = self.popupMenu.getPos(self.parent)
        scale = self.popupMenu.getScale(self.parent)
        # How are we doing relative to the right side of the screen
        maxX = pos[0] + fb[1] * scale[0]
        if maxX > 1.0:
            # Need to move menu to the left
            self.popupMenu.setX(self.parent, pos[0] + (1.0 - maxX))
        # How about up and down?
        minZ = pos[2] + fb[2] * scale[2]
        maxZ = pos[2] + fb[3] * scale[2]
        if minZ < -1.0:
            # Menu too low, move it up
            self.popupMenu.setZ(self.parent, pos[2] + (-1.0 - minZ))
        elif maxZ > 1.0:
            # Menu too high, move it down
            self.popupMenu.setZ(self.parent, pos[2] + (1.0 - maxZ))
        # Also display cancel frame to catch clicks outside of the popup
        self.cancelFrame.show()
        # Position and scale cancel frame to fill entire window
        self.cancelFrame.setPos(self.parent, 0, 0, 0)
        self.cancelFrame.setScale(self.parent, 1, 1, 1)

    def set(self, index, fCommand=True):
        """Set the new selected item.

        Parameters
        ----------
        index : int or string
            Index or label of the selected option.
        fCommand : bool, optional
            Whether to fire the selection callback or not. Default is True.
        """
        # Item was selected, record item and call command if any
        newIndex = self.index(index)
        if newIndex is not None:
            self.selectedIndex = newIndex
            item = self['items'][self.selectedIndex]
            if fCommand and self['command']:
                # Pass any extra args to command
                self['command'](*[item] + self['extraArgs'])
