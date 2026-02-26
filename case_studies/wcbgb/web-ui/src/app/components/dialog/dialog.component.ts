import { Clipboard } from '@angular/cdk/clipboard';
import { Component, Inject } from "@angular/core";
import { CommonModule } from "@angular/common";
import { MatButtonModule } from "@angular/material/button";
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from "@angular/material/dialog";
import { MatIconModule } from '@angular/material/icon';

type DialogData = {
  link: string,
  text: string,
  title: string,
};

@Component({
  selector: 'app-dialog',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatDialogModule,
    MatIconModule,
  ],
  templateUrl: './dialog.component.html',
  styleUrl: './dialog.component.scss'
})
export class DialogComponent {
  constructor(
    private clipboard: Clipboard,
    public dialogRef: MatDialogRef<DialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: DialogData
  ) {}

  close() {
    this.dialogRef.close();
  }

  copyLink() {
    this.clipboard.copy(this.data.link);
  }
}
