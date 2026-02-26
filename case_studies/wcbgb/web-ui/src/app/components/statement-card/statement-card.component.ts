import { OverlayModule } from '@angular/cdk/overlay';
import { CommonModule } from '@angular/common';
import { Component, Input, OnInit } from '@angular/core';
import { CustomTooltipDirective } from "../../directives/custom-tooltip/custom-tooltip.directive";

import { VoteGroup, Statement } from "../../models/report.model";

@Component({
  selector: 'app-statement-card',
  standalone: true,
  imports: [
    CommonModule,
    CustomTooltipDirective,
    OverlayModule,
  ],
  templateUrl: './statement-card.component.html',
  styleUrl: './statement-card.component.scss'
})
export class StatementCardComponent implements OnInit {
  @Input() data?: Statement;
  @Input() truncate = false;
  @Input() type = "";
  isOverallAgree?: boolean;
  agreePercent?: number;
  disagreePercent?: number;
  passPercent?: number;
  agreeTotal = 0;
  disagreeTotal = 0;
  passTotal = 0;
  voteTotal = 0;
  topics = "";

  ngOnInit() {
    if(!this.data) return;
    this.isOverallAgree = this.data.agreeRate >= this.data.disagreeRate;
    this.agreePercent = Math.round(this.data.agreeRate * 100);
    this.disagreePercent = Math.round(this.data.disagreeRate * 100);
    this.passPercent = Math.round(this.data.passRate * 100);
    Object.values(this.data.votes).forEach((voterGroup: VoteGroup) => {
      const { agreeCount, disagreeCount, passCount } = voterGroup;
      this.agreeTotal += agreeCount;
      this.disagreeTotal += disagreeCount;
      this.passTotal += passCount;
      this.voteTotal += agreeCount + disagreeCount + passCount;
    });
    if(this.data.topics) {
      this.topics = this.data.topics.replaceAll(";", ", ").replaceAll(":", " > ");
    }
  }
}
