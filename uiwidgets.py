"""
UI widgets.

NB: Naming conventions follow those used in other DirectObjects (camelCasing).
"""
from panda3d.core import TextNode
from panda3d.core import Vec3
import direct.gui.DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectButton
from direct.gui.DirectGui import DirectFrame
from direct.gui.DirectGui import DirectOptionMenu


class DropdownMenu(DirectOptionMenu):
    def __init__(self, parent=None, **kw):
        # We essentially reuse the __init__ from DirectOptionMenu but remove
        # the parts that are of no use for a drop-down menu.
        # NB: all things removed here only impacted setItems.

        # Define options. Why use this complicated system instead of a simple
        # list of keyword args? Some ideas:
        #  - when there is a LOT of keywords (as in e.g. matplotlib), this
        #  avoids bloating __init__ definitions while keeping the ability
        #  of defining default kw values and overriding them when inheriting.
        #  - this allows to attach each argument to a handler method in a
        #  coherent way.
        optiondefs = (
            # List of items to display on the popup menu
            ('items', [], self.setItems),
            # Background color to use to highlight popup menu items
            ('highlightColor', (.5, .5, .5, 1), None),
            # Extra scale to use on highlight popup menu items
            ('highlightScale', (1, 1), None),
            # Alignment to use for text on popup menu button
            ('text_align', TextNode.ACenter, None),
            # Remove press effect because it looks a bit funny
            ('pressEffect', 0, DGG.INITOPT),
           )
        # Merge keyword options with default options
        self.defineoptions(kw, optiondefs)
        # Initialize superclasses (the one we want here is DirectButton).
        DirectButton.__init__(self, parent)
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
        # Correct text position to obtain vertical centering with the button.
        for name in self.components():
            if "text" in name:
                text = self.component(name)
                text.setZ(self, text.node().getHeight() / (2*self['scale']))

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
                                              relief=self['relief'] or 'flat',
                                              )
        # Make sure it is on top of all the other gui widgets
        self.popupMenu.setBin('gui-popup', 0)
        if not self['items']:
            return
        # Create a new component for each item
        # Find the maximum extents of all items
        itemIndex = 0
        self.minX = self.maxX = self.minZ = self.maxZ = None
        # Reason why we use _constructorKeywords[*] and not self[*] for all the
        # 'text_*' options: see DirectGuiBase.py's docstring.
        # In a nutshell: __getitem__ only queries _optionInfo, to which
        # '*_*'-options are not added -- they are left in _constructorKeywords
        # instead, and consumed as they are used, UNLESS 'component' is a group
        # name, which 'text' is, because DirectFrame says so.
        for item in self['items']:
            c = self.createcomponent(
                'item{}'.format(itemIndex), (), 'item',
                DirectButton, (self.popupMenu,),
                text=item, text_align=TextNode.ALeft,
                text_font=self._constructorKeywords['text_font'][0],
                text_scale=self._constructorKeywords['text_scale'][0],
                #  text_fg=self._constructorKeywords['text_fg'][0],
                pad=(.3, .1),
                command=lambda i=itemIndex: self.set(i),
                relief=self['relief'] or 'flat',
                frameColor=self['frameColor'],
                )
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
            item = self.component('item{}'.format(i))
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
        self.popupMenu['frameSize'] = (
                0, self.maxWidth, -self.maxHeight * itemIndex, 0)
        # Set initial state
        self.hidePopupMenu()

    def showPopupMenu(self, event=None):
        """Make popup visible on top of the button.

        Adjust popup position if default position puts it outside of
        visible screen region.
        """
        # Show the menu
        self.popupMenu.show()
        # Make sure its at the right scale
        self.popupMenu.setScale(self, Vec3(1))
        # Compute bounds
        b = self.getBounds()
        fb = self.popupMenu.getBounds()
        # NB: the original coordinates of the menu are such that its top left
        # corner is at the origin.
        # Center menu with button horizontally
        xPos = (b[0] + b[1])/2 - (fb[1] - fb[0])/2.
        self.popupMenu.setX(self, xPos)
        # Set height slightly above the button
        margin = (b[3] - b[2]) * 0.2
        zPos = (b[2] + b[3])/2 + (b[3] - b[2])/2 + margin + (fb[3] - fb[2])
        self.popupMenu.setZ(self, zPos)
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

        Only difference with the original is that the text is not updated.

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


class ButtonMenu(DirectOptionMenu):
    """Simple menu with buttons."""
    def __init__(self, parent=None, **kw):
        optiondefs = (
            # List of items to display on the menu
            ('items', [], self.setItems),
            # Background color to use to highlight popup menu items
            ('highlightColor', (.5, .5, .5, 1), None),
            # Extra scale to use on highlight popup menu items
            ('highlightScale', (1, 1), None),
            # Command to be called on button click
            ('command',        None,       None),
            ('extraArgs',      [],         None),
            # Whether menu should be horizontal or vertical
            ('layout', 'horizontal', DGG.INITOPT),
            # Padding around the buttons
            ('pad', (.1, .1), DGG.INITOPT),
            )
        # Merge keyword options with default options
        self.defineoptions(kw, optiondefs)
        # Initialize the relevant superclass
        DirectFrame.__init__(self, parent)
        # Call option initialization functions
        self.initialiseoptions(ButtonMenu)

    def setItems(self):
        if not self['items']:
            return
        # Create a new component for each item
        # Find the maximum extents of all items
        itemIndex = 0
        self.minX = self.maxX = self.minZ = self.maxZ = None
        # Reason why we use _constructorKeywords[*] and not self[*] for all the
        # 'text_*' options: see DirectGuiBase.py's docstring.
        # In a nutshell: __getitem__ only queries _optionInfo, to which
        # '*_*'-options are not added -- they are left in _constructorKeywords
        # instead, and consumed as they are used, UNLESS 'component' is a group
        # name, which 'text' is, because DirectFrame says so.
        for item in self['items']:
            c = self.createcomponent(
                'item{}'.format(itemIndex), (), 'item',
                DirectButton, (self,),
                text=item, text_align=TextNode.ACenter,
                text_font=self._constructorKeywords['text_font'][0],
                text_scale=self._constructorKeywords['text_scale'][0],
                #  text_fg=self._constructorKeywords['text_fg'][0],
                pad=(.3, .2),
                command=lambda i=itemIndex: self.set(i),
                relief=self['relief'] or 'flat',
                borderWidth=(.01, .01),
                frameColor=self['frameColor'],
                )
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
            item = self.component('item{}'.format(i))
            # So entire extent of item's slot on popup is reactive to mouse
            item['frameSize'] = (self.minX, self.maxX, self.minZ, self.maxZ)
            # Move it to its correct position in the menu
            if self['layout'] == 'vertical':
                item.setPos(-self.minX, 0, -self.maxZ - i * self.maxHeight)
            else:
                item.setPos(-self.minX + i * self.maxWidth, 0, -self.maxZ)
            # Highlight background when mouse is in item
            item.bind(DGG.WITHIN,
                      lambda x, i=i, item=item: self._highlightItem(item, i))
            # Restore specified color upon exiting
            fc = item['frameColor']
            item.bind(DGG.WITHOUT,
                      lambda x, item=item, fc=fc: self._unhighlightItem(
                          item, fc))
        # Set popup menu frame size to encompass all items
        px, py = self['pad']
        self['frameSize'] = (
                -px, self.maxWidth * itemIndex + px, -self.maxHeight - py, py)

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
