"""Help > Quick start guide: the quick start guide, readable beside the app while you work.

The guide is ``docs/quick-start.md`` next to this file, the same Markdown that GitHub
shows, so one file is both the repository's guide and the app's. Qt renders it; this
module dresses it in the style guide's colours and fonts (headings in Barlow Semi
Condensed, uppercase, Gray 200; code and channel names in Roboto Mono) and adds a
contents list and a find box.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QPalette,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QTextFrameFormat,
    QTextLength,
    QTextTableFormat,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import theme as T

GUIDE = Path(__file__).with_name("docs") / "quick-start.md"


def guide_text() -> str:
    try:
        return GUIDE.read_text(encoding="utf-8")
    except OSError as e:
        return f"# Quick start guide\n\nThe guide could not be read from `{GUIDE}`: {e}"


def _heading_text(block) -> str:
    return block.text().strip()


def style_document(doc: QTextDocument) -> list[tuple[int, str, int]]:
    """Restyle a rendered guide in the app's theme. Returns its headings as (level, text, position)."""
    headings = []
    cur = QTextCursor(doc)
    cur.beginEditBlock()
    block = doc.begin()
    while block.isValid():
        level = block.blockFormat().headingLevel()
        bf = block.blockFormat()
        code_block = bf.hasProperty(QTextFormat.BlockCodeFence) or bf.nonBreakableLines()
        if level:
            headings.append((level, _heading_text(block), block.position()))
            f = QTextCharFormat()
            f.setFontFamilies([T.LABEL_FONT])
            f.setFontPointSize({1: 18, 2: 13, 3: 11}.get(level, 10))
            f.setFontWeight(QFont.DemiBold)
            f.setFontCapitalization(QFont.AllUppercase if level > 1 else QFont.MixedCase)
            f.setFontLetterSpacingType(QFont.AbsoluteSpacing)
            f.setFontLetterSpacing(1.0 if level > 1 else 0.5)
            f.setForeground(QBrush(QColor(T.TEXT if level == 1 else T.TEXT_DIM)))
            _apply(cur, block, f)
            bf.setTopMargin({1: 4, 2: 22, 3: 14}.get(level, 10))
            bf.setBottomMargin(6)
            c = QTextCursor(block)
            c.setBlockFormat(bf)
        elif code_block:
            bf.setBackground(QBrush(QColor(T.GRAY_950)))
            bf.setLeftMargin(8)
            bf.setRightMargin(8)
            QTextCursor(block).setBlockFormat(bf)
        for frag in _fragments(block):
            cf = frag.charFormat()
            if cf.fontFixedPitch() or "mono" in " ".join(cf.fontFamilies() or []).lower():
                f = QTextCharFormat()
                f.setFontFamilies([T.MONO_FONT])
                f.setFontFixedPitch(True)
                f.setFontPointSize(8.5)
                f.setForeground(QBrush(QColor(T.GRAY_100)))
                if not code_block:
                    f.setBackground(QBrush(QColor(T.GRAY_950)))
                c = QTextCursor(doc)
                c.setPosition(frag.position())
                c.setPosition(frag.position() + frag.length(), QTextCursor.KeepAnchor)
                c.mergeCharFormat(f)
            elif cf.isAnchor():
                f = QTextCharFormat()
                f.setForeground(QBrush(QColor(T.CYAN)))
                f.setFontUnderline(False)
                c = QTextCursor(doc)
                c.setPosition(frag.position())
                c.setPosition(frag.position() + frag.length(), QTextCursor.KeepAnchor)
                c.mergeCharFormat(f)
        block = block.next()
    _style_tables(doc)
    cur.endEditBlock()
    return headings


def _fragments(block):
    it = block.begin()
    while not it.atEnd():
        frag = it.fragment()
        if frag.isValid():
            yield frag
        it += 1


def _apply(cur: QTextCursor, block, fmt: QTextCharFormat) -> None:
    c = QTextCursor(block)
    c.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
    c.mergeCharFormat(fmt)


def _style_tables(doc: QTextDocument) -> None:
    """Tables: a thin Gray 600 grid, the header row on Gray 950 in Gray 200, full width."""
    seen = set()
    block = doc.begin()
    while block.isValid():
        table = QTextCursor(block).currentTable()
        if table is not None and id(table) not in seen and table.firstPosition() not in seen:
            seen.add(table.firstPosition())
            fmt: QTextTableFormat = table.format()
            fmt.setBorder(1)
            fmt.setBorderBrush(QBrush(QColor(T.GRAY_600)))
            fmt.setBorderStyle(QTextFrameFormat.BorderStyle_Solid)
            fmt.setBorderCollapse(True)
            fmt.setCellPadding(5)
            fmt.setCellSpacing(0)
            fmt.setWidth(QTextLength(QTextLength.PercentageLength, 100))
            fmt.setTopMargin(6)
            fmt.setBottomMargin(10)
            table.setFormat(fmt)
            for col in range(table.columns()):
                cell = table.cellAt(0, col)
                cf = cell.format()
                cf.setBackground(QBrush(QColor(T.GRAY_950)))
                cell.setFormat(cf)
                c = cell.firstCursorPosition()
                c.setPosition(cell.lastCursorPosition().position(), QTextCursor.KeepAnchor)
                f = QTextCharFormat()
                f.setForeground(QBrush(QColor(T.TEXT_DIM)))
                f.setFontFamilies([T.LABEL_FONT])
                f.setFontPointSize(10)
                f.setFontWeight(QFont.DemiBold)
                c.mergeCharFormat(f)
        block = block.next()


class GuideWindow(QWidget):
    """The quick start guide in its own window, with contents on the left and a find box."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("OwlFluids quick start guide")
        self.setObjectName("guide")

        self.find_box = QLineEdit()
        self.find_box.setPlaceholderText("Find in guide (Enter for next)")
        self.find_box.setClearButtonEnabled(True)
        self.find_box.returnPressed.connect(self.find_next)
        self.find_box.textChanged.connect(lambda _t: self.find_next(restart=True))
        self.find_status = QLabel("")
        self.find_status.setObjectName("hint")

        self.contents = QListWidget()
        self.contents.itemClicked.connect(self._go)
        self.contents.currentItemChanged.connect(lambda item, _prev: item is not None and self._go(item))

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setObjectName("guide_text")
        pal = self.browser.palette()
        pal.setColor(QPalette.Link, QColor(T.CYAN))
        pal.setColor(QPalette.LinkVisited, QColor(T.CYAN))
        self.browser.setPalette(pal)
        self.browser.document().setDocumentMargin(18)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        head = QLabel("CONTENTS")
        head.setObjectName("dock_title")
        ll.addWidget(head)
        ll.addWidget(self.contents, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.browser)
        split.setStretchFactor(1, 1)
        split.setSizes([230, 700])

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        bar.addWidget(self.find_box, 1)
        bar.addWidget(self.find_status)
        top = QWidget()
        top.setObjectName("guide_bar")
        top.setLayout(bar)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(top)
        root.addWidget(split, 1)

        QShortcut(QKeySequence.Find, self, activated=lambda: (self.find_box.setFocus(), self.find_box.selectAll()))
        QShortcut(QKeySequence("Escape"), self, activated=self.close)
        self.setStyleSheet(f"""
            QWidget#guide_bar {{ background: {T.GRAY_950}; border-bottom: 1px solid {T.BORDER}; }}
            QLabel#dock_title {{ background: {T.GRAY_950}; color: {T.TEXT_DIM}; padding: 5px 8px;
                                font-family: "{T.LABEL_FONT}"; font-size: 10pt; letter-spacing: 1px; }}
            QTextBrowser#guide_text {{ background: {T.PANEL}; border: none; }}
        """)
        self.load()
        self.resize(1000, 760)

    def load(self) -> None:
        doc = self.browser.document()
        doc.setMarkdown(guide_text())
        self.headings = style_document(doc)
        self.contents.blockSignals(True)
        self.contents.clear()
        for level, text, pos in self.headings:
            if level == 1:
                continue
            item = QListWidgetItem(("      " if level > 2 else "") + text)
            item.setData(Qt.UserRole, pos)
            if level == 2:
                f = T.ui_font(9, bold=True)
                item.setFont(f)
            else:
                item.setForeground(QBrush(QColor(T.TEXT_DIM)))
            self.contents.addItem(item)
        self.contents.setCurrentRow(-1)
        self.contents.blockSignals(False)
        self.browser.moveCursor(QTextCursor.Start)
        self.browser.verticalScrollBar().setValue(0)

    def _go(self, item: QListWidgetItem) -> None:
        self.show_position(int(item.data(Qt.UserRole)))

    def show_position(self, pos: int) -> None:
        """Scroll so the heading at ``pos`` sits at the top of the view."""
        block = self.browser.document().findBlock(pos)
        top = self.browser.document().documentLayout().blockBoundingRect(block).top()
        self.browser.verticalScrollBar().setValue(int(top))
        c = QTextCursor(block)
        self.browser.setTextCursor(c)

    def show_section(self, title: str) -> bool:
        """Scroll to the first heading containing ``title`` (case ignored). False if there is none."""
        for _level, text, pos in self.headings:
            if title.lower() in text.lower():
                self.show_position(pos)
                return True
        return False

    def find_next(self, restart: bool = False) -> None:
        text = self.find_box.text()
        if not text:
            self.find_status.setText("")
            return
        if restart:
            c = self.browser.textCursor()
            c.setPosition(c.selectionStart() if c.hasSelection() else 0)
            if not c.hasSelection():
                c.movePosition(QTextCursor.Start)
            self.browser.setTextCursor(c)
        if self.browser.find(text):
            self.find_status.setText("")
            return
        self.browser.moveCursor(QTextCursor.Start)  # wrap around once
        self.find_status.setText("" if self.browser.find(text) else "Not found")
