import { CommonModule } from '@angular/common';
import { Component, CUSTOM_ELEMENTS_SCHEMA, QueryList, ViewChildren } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatDialog, MatDialogModule } from "@angular/material/dialog";
import { MatExpansionModule, MatExpansionPanel } from '@angular/material/expansion';
import { MatIconModule } from '@angular/material/icon';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MarkdownModule } from 'ngx-markdown';
import { DialogComponent } from '../../components/dialog/dialog.component';
import { SensemakingChartWrapperComponent } from '../../components/sensemaking-chart-wrapper/sensemaking-chart-wrapper.component';
import { StatementCardComponent } from '../../components/statement-card/statement-card.component';

import importedTopicData from  "../../../../data/topic-stats.json";
import importedSummaryData from  "../../../../data/summary.json";
import importedCommentData from  "../../../../data/comments.json";
import importedReportMetadata from "../../../../data/metadata.json";

import {
  VoteGroup,
  Statement,
  Subtopic,
  Topic,
} from "../../models/report.model";

type AlignmentType = "high-alignment" | "low-alignment" | "high-uncertainty";

let totalVoteNumber = 0;

// use comment/topic(subtopic) relationship in comment data to add each comment's details to the topic configuration object
// also adding up each comment's votes to get total votes for report
importedCommentData.forEach((comment: Statement) => {
  const commentTopics = comment.topics?.split(";");
  commentTopics?.forEach((topicSubtopicString: string) => {
    const [topic, subtopic] = topicSubtopicString.split(":");

    // add comment to each of its subtopics in the topic data
    const configTopic = importedTopicData.find((topicData: Topic) => topicData.name === topic);
    const configSubtopic: Subtopic|undefined = configTopic?.subtopicStats?.find((subtopicData: Subtopic) => subtopicData.name === subtopic);
    if(!configSubtopic) return;
    if(!configSubtopic.comments) {
      configSubtopic.comments = [comment];
    } else {
      configSubtopic.comments.push(comment);
    }
  });

  // add comment votes to running vote total
  const voteGroups = Object.values(comment.votes || {});
  const commentVotes: number = voteGroups.reduce((commentAcc: number, group: VoteGroup) => {
    return commentAcc + group.agreeCount + group.disagreeCount + group.passCount;
  }, 0);
  totalVoteNumber += commentVotes;
});

const allSubtopicIds: string[] = [];
importedTopicData.forEach((t: Topic) => {
  t.subtopicStats.forEach((s: Subtopic) => {
    const subtopicId = `${t.name}-${s.name}`;
    s.id = subtopicId;
    allSubtopicIds.push(subtopicId);
  });
});

@Component({
  selector: 'app-report',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MarkdownModule,
    MatButtonModule,
    MatButtonToggleModule,
    MatDialogModule,
    MatExpansionModule,
    MatIconModule,
    MatSidenavModule,
    MatTooltipModule,
    SensemakingChartWrapperComponent,
    StatementCardComponent,
  ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  templateUrl: './report.component.html',
  styleUrl: './report.component.scss'
})
export class ReportComponent {
  // data sources
  topicData = importedTopicData as Topic[];
  summaryData = importedSummaryData;
  commentData = importedCommentData;
  reportMetadata = importedReportMetadata;

  reportTitle: string = this.reportMetadata.title || "Report";
  selectedAlignmentType: AlignmentType = "high-alignment";
  isStatementDrawerOpen = false;
  drawerSubtopicName = "";
  drawerSubtopicStatementNumber = 0;
  drawerSubtopicStatementsHighAlignment: Statement[] = [];
  drawerSubtopicStatementsLowAlignment: Statement[] = [];
  drawerSubtopicStatementsHighUncertainty: Statement[] = [];
  drawerSubtopicStatementsUncategorized: Statement[] = [];

  topicAlignmentViews: { [key: string]: string } = {};
  topicsDistributionView = 'cluster';

  @ViewChildren("subtopicPanel") subtopicPanels!: QueryList<MatExpansionPanel>;

  constructor(private dialog: MatDialog) {}

  ngOnInit(): void {
    // Initialize view states for each topic
    this.topicData.forEach((topic: { name: string }) => {
      this.topicAlignmentViews[topic.name] = 'solid';
    });
  }

  updateTopicView(topicName: string, view: string): void {
    this.topicAlignmentViews[topicName] = view;
  }

  openShareReportDialog({
    elementId,
    text,
    title,
  }: {
    elementId?: string,
    text: string,
    title: string,
  }) {
    let link = window.location.origin + window.location.pathname;
    if(elementId) {
      link += `#${encodeURIComponent(elementId)}`;
    }
    this.dialog.open(DialogComponent, {
      data: {
        link,
        text,
        title,
      }
    });
  }

  scrollToElement(elementId?: string) {
    const element = document.getElementById(elementId || "");
    element?.scrollIntoView({ behavior: "smooth" });
  }

  // used by nav to open subtopic accordion panel
  // scrolling to subtopic is handled by event listener responding to "expand" event
  openSubtopicPanel(elementId?: string) {
    const panelIndex = allSubtopicIds.indexOf(elementId || "");
    this.subtopicPanels.toArray()[panelIndex]?.open();
  }

  // triggering scroll following opening of subtopic panel by 1) nav and by 2) direct accordion interaction
  afterSubtopicPanelOpen(elementId?: string) {
    this.scrollToElement(elementId);
  }

  topicNumber = this.topicData.length;
  subtopicNumber = allSubtopicIds.length;
  totalStatements = this.commentData.length;
  totalVotes = totalVoteNumber;

  get alignmentString() {
    switch(this.selectedAlignmentType) {
      case "high-alignment":
        return "highest alignment";
      case "low-alignment":
        return "lowest alignment";
      case "high-uncertainty":
        return "highest uncertainty";
      default:
        return "";
    }
  }

  getTopStatements(statements: Statement[], category: AlignmentType): Statement[] {
    // no need to remove "isFilteredOut" statements first since "isFilteredOut" is exclusive to the 3 categories
    switch(category) {
      case "high-alignment":
        return statements
          .filter((statement: Statement) => statement.isHighAlignment)
          .sort((a: Statement, b: Statement) => b.highAlignmentScore - a.highAlignmentScore)
          .slice(0, 12);
      case "low-alignment":
        return statements
          .filter((statement: Statement) => statement.isLowAlignment)
          .sort((a: Statement, b: Statement) => b.lowAlignmentScore - a.lowAlignmentScore)
          .slice(0, 12);
      case "high-uncertainty":
        return statements
          .filter((statement: Statement) => statement.isHighUncertainty)
          .sort((a: Statement, b: Statement) => b.highUncertaintyScore - a.highUncertaintyScore)
          .slice(0, 12);
      default:
        return [];
    }
  }

  get alignmentCards() {
    return this.getTopStatements(this.commentData, this.selectedAlignmentType);
  }

  getTopicSummaryData(topicName: string): any {
    const summaryTopicData: any = this.summaryData.contents.find(c => c.title.includes("Topics"));
    const topicData = summaryTopicData?.subContents.find((s: any) => s.title.includes(topicName));
    return topicData;
  }

  getSubtopicSummaryData(topicName: string, subtopicName: string): any {
    const topicData = this.getTopicSummaryData(topicName);
    const subtopicData = topicData?.subContents.find((s: any) => s.title.includes(subtopicName));
    return subtopicData;
  }

  getSubtopicThemesData(topicName: string, subtopicName: string): any {
    const subtopicData = this.getSubtopicSummaryData(topicName, subtopicName);
    const subtopicThemesData = subtopicData?.subContents.find((s: any) => s.title.includes("themes"));
    return subtopicThemesData;
  }

  getSubtopicThemesText(topicName: string, subtopicName: string): string {
    const subtopicThemesData = this.getSubtopicThemesData(topicName, subtopicName);
    return subtopicThemesData?.text || "";
  }

  getSubtopicStatements(topicName: string, subtopicName: string): Statement[] {
    const topic: Topic|undefined = this.topicData.find(t => t.name === topicName);
    const subtopic = topic?.subtopicStats.find((s: Subtopic) => s.name === subtopicName);
    return subtopic?.comments || [];
  }

  getTopSubtopicStatements(topicName: string, subtopicName: string, category: AlignmentType): Statement[] {
    const subtopicStatements = this.getSubtopicStatements(topicName, subtopicName);
    return this.getTopStatements(subtopicStatements, category);
  }

  openStatementDrawer(subtopic: Subtopic) {
    this.drawerSubtopicName = subtopic.name;
    this.drawerSubtopicStatementNumber = subtopic.commentCount;
    const highAlignmentStatements: Statement[] = [];
    const lowAlignmentStatements: Statement[] = [];
    const highUncertaintyStatements: Statement[] = [];
    const uncategorizedStatements: Statement[] = [];
    subtopic.comments?.forEach((statement: Statement) => {
      if(statement.isHighAlignment) {
        highAlignmentStatements.push(statement);
      } else if(statement.isLowAlignment) {
        lowAlignmentStatements.push(statement);
      } else if(statement.isHighUncertainty) {
        highUncertaintyStatements.push(statement);
      } else {
        uncategorizedStatements.push(statement);
      }
    });
    this.drawerSubtopicStatementsHighAlignment = highAlignmentStatements.sort((a: Statement, b: Statement) => b.highAlignmentScore - a.highAlignmentScore);
    this.drawerSubtopicStatementsLowAlignment = lowAlignmentStatements.sort((a: Statement, b: Statement) => b.lowAlignmentScore - a.lowAlignmentScore);
    this.drawerSubtopicStatementsHighUncertainty = highUncertaintyStatements.sort((a: Statement, b: Statement) => b.highUncertaintyScore - a.highUncertaintyScore);
    this.drawerSubtopicStatementsUncategorized = uncategorizedStatements.sort((a: Statement, b: Statement) => b.agreeRate - a.agreeRate);
    this.isStatementDrawerOpen = true;
  }

  closeStatementDrawer() {
    this.isStatementDrawerOpen = false;
    this.drawerSubtopicName = "";
    this.drawerSubtopicStatementNumber = 0;
    // clearing statement arrays also ensures that scroll position is reset to top
    this.drawerSubtopicStatementsHighAlignment = [];
    this.drawerSubtopicStatementsLowAlignment = [];
    this.drawerSubtopicStatementsHighUncertainty = [];
    this.drawerSubtopicStatementsUncategorized = [];
  }
}
